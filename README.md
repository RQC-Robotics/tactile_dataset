### UR5e tactile dataset.
This repo describes a tactile dataset collection.
The dataset contains tactile sensors' feedback  on performing continuous grasping of various household items alongside with information from cameras and robot.

AWS CLI can be used to synchronize the dataset with Yandex Object Storage:
```
aws s3 --endpoint-url=https://storage.yandexcloud.net sync local_dataset_path/ s3://tactile-dataset/dataset
```
There is also `s3://tactile-dataset/dataset.zip` that contains all the data but may be slightly stale.

Content:
- [collection.py](collection.py): the collection loop implementation.

- [_nodes.py](_nodes.py): provides custom task-specific nodes. 
