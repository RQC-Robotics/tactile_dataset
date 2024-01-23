## UR5e tactile dataset.
### Description
This repo describes a tactile dataset collection.
The dataset contains tactile sensors' feedback 
on performing continuous grasping of various household items
alongside with information from a camera and a robot.

An item of the dataset is a `dict` of numpy arrays:

| Key             | Value shape      | Value dtype | 
| :-------------- | :--------------- | :---------- |
| pos             | (T, 1)           | uint8       |
| object_detected | (T, 1)           | bool        |
| left_sensor     | (T, 320, 240, 3) | uint8       |
| right_sensor    | (T, 320, 240, 3) | uint8       |
| wrist_camera    | (T, 480, 640, 3) | uint8       |
| obj_hardness    | ()               | int64       |

where T - is a size of the Time dimension that is different for every object.

### Download
AWS CLI can be used to synchronize the dataset with Yandex Object Storage:
```
aws s3 --endpoint-url=https://storage.yandexcloud.net sync local_dataset_path/ s3://tactile-dataset/dataset
```
There is also `s3://tactile-dataset/dataset.zip` that contains all the data but may be slightly stale.

### The repository contents
- [collection.py](collection.py): the collection loop implementation.

- [_nodes.py](_nodes.py): provides custom task-specific nodes. 
