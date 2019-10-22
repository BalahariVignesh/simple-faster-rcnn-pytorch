docker run \
--name pytorch \
--gpus all \
-it \
-p 8890:8890 \
-v /home/tadenoud/:/workspace/tadenoud/ \
-v /media/tadenoud/DATADisk/datasets/:/workspace/datasets/ \
pytorch/pytorch:latest
