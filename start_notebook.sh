# please use the latest master of analytics-zoo

# please install intel-tensorflow==1.12.0, pip install intel-tensorflow==1.12.0

# please download darknet_keras at https://drive.google.com/file/d/1eeUyhBVEYnZpv0ICE5BTZh-YwZHbj09z/view?usp=sharing to ./models

export SPARK_HOME=/home/arda/bozhou/spark-2.1.1-bin-hadoop2.7
export ANALYTICS_ZOO_HOME=/home/arda/bozhou/dist
export ZOO_NUM_MKLTHREADS=18

sh $ANALYTICS_ZOO_HOME/bin/jupyter-with-zoo.sh
