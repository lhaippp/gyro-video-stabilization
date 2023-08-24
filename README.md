# gyro-video-stabilization

1. convert raw gyro data into homography
```
python gyro_readings_to_H.py --project_path . --filename 20200109_145447 --idx 100 200 
```

2. image alignment with the gyro-homo
```
python gyro_aglinment.py --data_path 20200109_145447 --idx 100 200 --split RE --gif_path gifs
```
