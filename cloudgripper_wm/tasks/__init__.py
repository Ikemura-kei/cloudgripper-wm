from cloudgripper_wm.tasks.base import Task
from cloudgripper_wm.tasks.cube_push import CubePushTask
from cloudgripper_wm.tasks.cube_stack import CubeStackTask
from cloudgripper_wm.tasks.rope_manip import RopeManipTask

TASK_REGISTRY: dict[str, type[Task]] = {
    "cube_push": CubePushTask,
    "cube_stack": CubeStackTask,
    "rope_manip": RopeManipTask,
}


def get_task(task_name: str | None) -> Task | None:
    if task_name is None:
        return None
    return TASK_REGISTRY[task_name]()
