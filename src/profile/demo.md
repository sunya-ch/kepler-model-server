#  Demo Steps

1. monitoring system is deployed

kubectl get po -n monitoring

2. clone three repository

git clone git@github.com:sustainable-computing-io/kepler.git
git clone git@github.com:sunya-ch/kepler-model-server.git
git clone git@github.com:sunya-ch/kepler-model-db.git

3. deploy kepler with prometheus

cd kepler
make build-manifest OPTS="PROMETHEUS_DEPLOY"
kubectl apply -f _output/generated-manifest/deployment.yaml

4. test kepler metrics

cd kepler-model-server/src/profile
git checkout new-pipeline-demo
./script.sh port_forward
curl localhost:30090/api/v1/query?query=up|grep kepler-exporter
kill $(pidof kubectl)

1. python environment is ready

conda activate kepler-model-server
pip freeze|grep prometheus

4. run 
./script.sh run
- check requisite
- deploy cpe operator
- deploy benchmark
- wait for bechmark to complete
- save
- commit log

5. push 
./script.sh push