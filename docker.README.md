# Docker README

## Build image.

`docker build --progress=plain -f Dockerfile -t the-point-is-the-mask:latest . 2>&1 | tee build.log`

`docker build -f Dockerfile -t the-point-is-the-mask:latest .`

## Launch image


```bash
docker run --gpus all --rm \
-v /media/bioeos/E/drone_serge_test/:/home/seatizen/plancha \
-v ./models:/home/seatizen/app/models \
-v /home/bioeos/.cache/huggingface:/home/seatizen/.cache/huggingface \
--name drone-inference drone-inference:latest -c -eses -pses /home/seatizen/plancha/20231201_REU-HERMITAGE_UAV_01
```
