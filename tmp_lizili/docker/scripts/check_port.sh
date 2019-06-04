DOCKER_NAME=${USER}_roadstar_dev
if [ -z "$(docker ps | grep ${DOCKER_NAME})" ]; then
  echo "${DOCKER_NAME} does NOT exist and run dev_start first to start a docker!"
else 
  docker port ${DOCKER_NAME} | sed "s/2222[^:]*/ssh/" | sed "s/8888[^:]*/dreamview/" | sed "s/6060[^:]*/simulation_world/" | sed "s/8443[^:]*/vscode/"
fi
