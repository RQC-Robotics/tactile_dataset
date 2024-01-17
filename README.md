### UR5e tactile dataset.
This repo describes tactile dataset collection.
A dataset contains tactile sensors' feedback alongside with information from cameras and robot (or any other node that is compatible with the [ur_env](https://github.com/RQC-Robotics/ur5-env.git))
 when performing continuous grasping of various items.

Dataset can be downloaded via preconfigured AWS CLI:
```
aws s3 --endpoint-url=https://storage.yandexcloud.net sync s3://tactile-dataset local_path/
```

Content:
- [collection.py](collection.py): the collection loop implementation.

- [_nodes.py](_nodes.py): provides custom task-specific nodes. 
