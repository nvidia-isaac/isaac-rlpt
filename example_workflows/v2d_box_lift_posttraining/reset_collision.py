# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh reset-pose collision queries using PhysX-cooked convex geometry."""

from dataclasses import dataclass

import fcl
import numpy as np
import omni.physx
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat
from numpy.typing import NDArray
from omni.physics.tensors.impl.api import RigidBodyView
from omni.physx.bindings._physx import (
    PhysxCollisionRepresentationResult,
    PhysxConvexMeshData,
)
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

from example_workflows.v2d_box_lift_posttraining.reset_config import (
    ResetValidationConfig,
)
from example_workflows.v2d_box_lift_posttraining.reset_validation import Penetration


@dataclass
class CollisionShape:
    geometry: fcl.CollisionGeometry
    local_transform: NDArray[np.float64]


@dataclass
class CollisionBody:
    name: str
    path: str
    asset_name: str | None
    body_index: int
    fixed_transform: NDArray[np.float64]
    shapes: list[CollisionShape]
    global_body: bool
    live_view: RigidBodyView | None
    view_indices: tuple[int, ...]


def rigid_transform(matrix: Gf.Matrix4d) -> NDArray[np.float64]:
    result = np.eye(4)
    result[:3, :3] = np.asarray(matrix.ExtractRotationMatrix()).T
    # USD matrices use row vectors; FCL uses column vectors.
    lengths = np.linalg.norm(result[:3, :3], axis=0)
    result[:3, :3] /= lengths
    result[:3, 3] = matrix.ExtractTranslation()
    return result


def _isolated_mesh(
    prim: Usd.Prim, collider: Usd.Prim
) -> tuple[Usd.Stage, UsdGeom.Mesh]:
    """An in-memory copy of ``prim`` carrying ``collider``'s collision settings."""
    cooking_stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(cooking_stage, "/Mesh")
    source = UsdGeom.Mesh(prim)
    mesh.CreatePointsAttr(source.GetPointsAttr().Get())
    mesh.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    mesh.CreateOrientationAttr(source.GetOrientationAttr().Get())
    schemas = list(
        dict.fromkeys([*collider.GetAppliedSchemas(), *prim.GetAppliedSchemas()])
    )
    for schema in schemas:
        mesh.GetPrim().AddAppliedSchema(schema)
    for settings in (collider, prim):
        _copy_physics_attributes(settings, mesh.GetPrim())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
    )
    return cooking_stage, mesh


def _copy_physics_attributes(source: Usd.Prim, target: Usd.Prim) -> None:
    for attribute in source.GetAttributes():
        if not attribute.GetName().startswith(("physics:", "physx")):
            continue
        value = attribute.Get()
        if value is not None:
            target.CreateAttribute(attribute.GetName(), attribute.GetTypeName()).Set(
                value
            )


def _fcl_convex(
    convex: PhysxConvexMeshData, transform: NDArray[np.float64]
) -> fcl.Convex:
    vertices = np.asarray([[p.x, p.y, p.z] for p in convex.vertices])
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    faces: list[int] = []
    for polygon in convex.polygons:
        faces.append(polygon.num_vertices)
        faces.extend(
            convex.indices[
                polygon.index_base : polygon.index_base + polygon.num_vertices
            ]
        )
    return fcl.Convex(vertices, len(convex.polygons), np.asarray(faces, dtype=np.int32))


def cooked_convexes(
    prim: Usd.Prim, collider: Usd.Prim, transform: NDArray[np.float64]
) -> list[fcl.Convex]:
    result_shapes: list[fcl.Convex] = []
    outcomes: list[PhysxCollisionRepresentationResult] = []
    # Imported URDFs author collision APIs on an Xform above the mesh. Cook an
    # isolated mesh with those exact settings; never modify the simulation stage.
    cooking_stage, mesh = _isolated_mesh(prim, collider)
    stage_id = UsdUtils.StageCache.Get().Insert(cooking_stage)

    def receive(
        result: PhysxCollisionRepresentationResult, convexes: list[PhysxConvexMeshData]
    ) -> None:
        outcomes.append(result)
        if result != PhysxCollisionRepresentationResult.RESULT_VALID:
            return
        result_shapes.extend(_fcl_convex(convex, transform) for convex in convexes)

    try:
        omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id.ToLongInt(),
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(mesh.GetPath()),
            run_asynchronously=False,
            on_result=receive,
        )
    finally:
        UsdUtils.StageCache.Get().Erase(stage_id)
    if (
        outcomes != [PhysxCollisionRepresentationResult.RESULT_VALID]
        or not result_shapes
    ):
        raise RuntimeError(f"Collision cooking failed for {prim.GetPath()}: {outcomes}")
    return result_shapes


def _mesh_shapes(
    prim: Usd.Prim, transform: NDArray[np.float64]
) -> list[CollisionShape]:
    approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
    if approximation not in ("convexHull", "convexDecomposition"):
        raise ValueError(
            "Reset validation requires convex collision meshes: "
            f"{prim.GetPath()} ({approximation})"
        )
    shapes: list[CollisionShape] = []
    cache = UsdGeom.XformCache()
    root_inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(prim)).T)
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child.IsA(UsdGeom.Mesh):
            local = (
                transform
                @ root_inverse
                @ np.asarray(cache.GetLocalToWorldTransform(child)).T
            )
            shapes.extend(
                CollisionShape(geometry, np.eye(4))
                for geometry in cooked_convexes(child, prim, local)
            )
    if not shapes:
        raise ValueError(f"Collision root contains no meshes: {prim.GetPath()}")
    return shapes


def _cylinder_shape(prim: Usd.Prim, transform: NDArray[np.float64]) -> CollisionShape:
    cylinder = UsdGeom.Cylinder(prim)
    axis = str(cylinder.GetAxisAttr().Get())
    rotations = {
        "X": np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "Y": np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        "Z": np.eye(3),
    }
    aligned = transform.copy()
    aligned[:3, :3] = transform[:3, :3] @ rotations[axis]
    scales = np.linalg.norm(aligned[:3, :3], axis=0)
    aligned[:3, :3] /= scales
    if not np.isclose(scales[0], scales[1]) or not np.allclose(
        aligned[:3, :3].T @ aligned[:3, :3], np.eye(3)
    ):
        raise ValueError(
            f"Unsupported noncircular or sheared cylinder: {prim.GetPath()}"
        )
    geometry = fcl.Cylinder(
        float(cylinder.GetRadiusAttr().Get()) * scales[0],
        float(cylinder.GetHeightAttr().Get()) * scales[2],
    )
    return CollisionShape(geometry, aligned)


def collision_shapes(
    prim: Usd.Prim, transform: NDArray[np.float64]
) -> list[CollisionShape]:
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        return _mesh_shapes(prim, transform)
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
        vertices = np.asarray(
            [
                [-half, -half, -half],
                [half, -half, -half],
                [half, half, -half],
                [-half, half, -half],
                [-half, -half, half],
                [half, -half, half],
                [half, half, half],
                [-half, half, half],
            ]
        )
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        faces: NDArray[np.int32] = np.asarray(
            [
                4,
                0,
                3,
                2,
                1,
                4,
                4,
                5,
                6,
                7,
                4,
                0,
                1,
                5,
                4,
                4,
                1,
                2,
                6,
                5,
                4,
                2,
                3,
                7,
                6,
                4,
                3,
                0,
                4,
                7,
            ],
            dtype=np.int32,
        )
        return [CollisionShape(fcl.Convex(vertices, 6, faces), np.eye(4))]
    if prim.IsA(UsdGeom.Cylinder):
        return [_cylinder_shape(prim, transform)]
    if prim.IsA(UsdGeom.Plane):
        axis = str(UsdGeom.Plane(prim).GetAxisAttr().Get())
        normals = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
        normal = np.linalg.inv(transform[:3, :3]).T @ np.asarray(normals[axis])
        normal /= np.linalg.norm(normal)
        return [
            CollisionShape(
                fcl.Halfspace(normal, float(normal @ transform[:3, 3])), np.eye(4)
            )
        ]
    raise ValueError(
        f"Unsupported reset collision shape: {prim.GetPath()} ({prim.GetTypeName()})"
    )


def _asset_roots(env: ManagerBasedRLEnv) -> dict[str, tuple[str, int]]:
    """Map each articulation link and rigid-object root path to (asset name, body index)."""
    roots: dict[str, tuple[str, int]] = {}
    for name, articulation in env.scene.articulations.items():
        for index, path in enumerate(articulation.root_physx_view.link_paths[0]):
            roots[path] = (name, index)
    for name, asset in env.scene.rigid_objects.items():
        roots[asset.root_physx_view.prim_paths[0]] = (name, 0)
    if env.scene.rigid_object_collections:
        raise ValueError(
            "Reset collision validation does not support rigid-object collections"
        )
    return roots


def _owning_body(
    prim: Usd.Prim, roots: dict[str, tuple[str, int]], env_path: str
) -> Usd.Prim:
    """The nearest ancestor that is an asset root or a rigid body, else the topmost one below the env."""
    body_prim = prim
    while body_prim.GetPath().pathString not in roots and not body_prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        parent = body_prim.GetParent()
        if parent.IsPseudoRoot() or parent.GetPath().pathString in (
            env_path,
            "/World",
        ):
            break
        body_prim = parent
    return body_prim


def _fcl_objects(
    body: CollisionBody, transform: NDArray[np.float64]
) -> list[fcl.CollisionObject]:
    objects: list[fcl.CollisionObject] = []
    for shape in body.shapes:
        placement = transform @ shape.local_transform
        objects.append(
            fcl.CollisionObject(
                shape.geometry, fcl.Transform(placement[:3, :3], placement[:3, 3])
            )
        )
    return objects


def _deepest_penetration(
    firsts: list[fcl.CollisionObject],
    seconds: list[fcl.CollisionObject],
    request: fcl.CollisionRequest,
) -> float:
    depth = 0.0
    for first_shape in firsts:
        for second_shape in seconds:
            result = fcl.CollisionResult()
            fcl.collide(first_shape, second_shape, request, result)
            for contact in result.contacts:
                if not np.isfinite(contact.penetration_depth):
                    raise ValueError(
                        "Collision query returned a non-finite penetration"
                    )
                depth = max(depth, float(contact.penetration_depth))
    return depth


class ResetCollisionChecker:
    """Cache geometry once, then read fresh poses after every candidate reset."""

    def __init__(self, env: ManagerBasedRLEnv, config: ResetValidationConfig) -> None:
        self.env = env
        self.config = config
        stage = env.sim.stage
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError("Reset collision validation requires meter-scale USD")
        env_path = env.scene.env_prim_paths[0]
        self.excluded_pairs: set[frozenset[str]] = set()
        self.bodies = self._collect_bodies(stage, env_path)
        names = {body.name for body in self.bodies}
        for pair in config.allowed_contact_pairs:
            if not set(pair) <= names:
                raise ValueError(
                    f"Unknown allowed contact pair {pair}; "
                    f"available bodies: {sorted(names)}"
                )
        self.allowed_pairs = {frozenset(pair) for pair in config.allowed_contact_pairs}
        if len(self.bodies) < 2 or not any(
            body.asset_name == "robot" for body in self.bodies
        ):
            raise ValueError(
                "Reset validation did not find robot and scene collision geometry"
            )

    def _collect_bodies(self, stage: Usd.Stage, env_path: str) -> list[CollisionBody]:
        """Group the collision prims of the first environment and the shared scene by body."""
        roots = _asset_roots(self.env)
        bodies: list[CollisionBody] = []
        by_path: dict[str, CollisionBody] = {}
        cache = UsdGeom.XformCache()
        prims = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        for prim in prims:
            prim_path = str(prim.GetPath())
            if prim_path.startswith("/World/envs/") and not (
                prim_path == env_path or prim_path.startswith(env_path + "/")
            ):
                continue
            self._exclude_joint_pair(prim)
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            body_prim = _owning_body(prim, roots, env_path)
            path = str(body_prim.GetPath())
            if path not in by_path:
                body = self._new_body(body_prim, roots, env_path, cache)
                by_path[path] = body
                bodies.append(body)
            body = by_path[path]
            local = (
                np.linalg.inv(body.fixed_transform)
                @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            )
            body.shapes.extend(collision_shapes(prim, local))
        return bodies

    def _exclude_joint_pair(self, prim: Usd.Prim) -> None:
        """Record the bodies of a joint that disables collision between them."""
        if (
            not prim.IsA(UsdPhysics.Joint)
            or UsdPhysics.Joint(prim).GetCollisionEnabledAttr().Get()
        ):
            return
        joint = UsdPhysics.Joint(prim)
        for first in joint.GetBody0Rel().GetTargets():
            for second in joint.GetBody1Rel().GetTargets():
                self.excluded_pairs.add(frozenset((str(first), str(second))))

    def _new_body(
        self,
        body_prim: Usd.Prim,
        roots: dict[str, tuple[str, int]],
        env_path: str,
        cache: UsdGeom.XformCache,
    ) -> CollisionBody:
        path = str(body_prim.GetPath())
        pose = rigid_transform(cache.GetLocalToWorldTransform(body_prim))
        live_view = None
        view_indices: tuple[int, ...] = ()
        if path in roots:
            asset_name, index = roots[path]
            name = f"{asset_name}/{body_prim.GetName()}"
        elif body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            asset_name, index = None, 0
            name = body_prim.GetName()
            live_view, view_indices = self._live_view(path, env_path)
        else:
            asset_name, index = None, 0
            name = body_prim.GetName()
        return CollisionBody(
            name,
            path,
            asset_name,
            index,
            pose,
            [],
            not path.startswith(env_path + "/"),
            live_view,
            view_indices,
        )

    def _live_view(
        self, path: str, env_path: str
    ) -> tuple[RigidBodyView, tuple[int, ...]]:
        """A physics view of a non-asset rigid body and its index for each environment."""
        env = self.env
        pattern = path.replace(env_path, "/World/envs/*")
        live_view = env.sim.physics_sim_view.create_rigid_body_view(pattern)
        expected_count = env.num_envs if path.startswith(env_path + "/") else 1
        if live_view.count != expected_count:
            raise ValueError(f"Cannot resolve live collision body {path}")
        lookup = {name: i for i, name in enumerate(live_view.prim_paths)}
        view_indices = tuple(
            lookup[path.replace(env_path, selected_path)]
            for selected_path in env.scene.env_prim_paths
        )
        return live_view, view_indices

    def _body_transform(
        self, body: CollisionBody, env_index: int
    ) -> NDArray[np.float64]:
        """The body's current world pose in ``env_index``."""
        if body.live_view is not None:
            pose = body.live_view.get_transforms()[body.view_indices[env_index]]
            transform = np.eye(4)
            transform[:3, :3] = matrix_from_quat(pose[[6, 3, 4, 5]]).cpu().numpy()
            transform[:3, 3] = pose[:3].cpu().numpy()
        elif body.asset_name is None:
            transform = body.fixed_transform.copy()
            if not body.global_body:
                offset = (
                    self.env.scene.env_origins[env_index]
                    - self.env.scene.env_origins[0]
                )
                transform[:3, 3] += offset.cpu().numpy()
        else:
            pose = self.env.scene[body.asset_name].data.body_link_pose_w[
                env_index, body.body_index
            ]
            transform = np.eye(4)
            transform[:3, :3] = matrix_from_quat(pose[3:7]).cpu().numpy()
            transform[:3, 3] = pose[:3].cpu().numpy()
        if not np.isfinite(transform).all():
            raise ValueError(f"Non-finite reset pose for {body.name}")
        return transform

    def _pair_is_checked(self, first: CollisionBody, second: CollisionBody) -> bool:
        """Skip static-static, joint-excluded and explicitly allowed pairs."""
        both_static = (
            first.asset_name is None
            and first.live_view is None
            and second.asset_name is None
            and second.live_view is None
        )
        return not (
            both_static
            or frozenset((first.path, second.path)) in self.excluded_pairs
            or frozenset((first.name, second.name)) in self.allowed_pairs
        )

    def check(self, env_index: int) -> list[Penetration]:
        # Forward kinematics updates link poses without advancing physics or time.
        self.env.sim.forward()
        objects = [
            _fcl_objects(body, self._body_transform(body, env_index))
            for body in self.bodies
        ]
        invalid: list[Penetration] = []
        request = fcl.CollisionRequest(num_max_contacts=128, enable_contact=True)
        for first_index, first in enumerate(self.bodies):
            for second_index in range(first_index + 1, len(self.bodies)):
                second = self.bodies[second_index]
                if not self._pair_is_checked(first, second):
                    continue
                depth = _deepest_penetration(
                    objects[first_index], objects[second_index], request
                )
                if depth > self.config.max_penetration_m:
                    invalid.append(Penetration(first.name, second.name, depth))
        return invalid
