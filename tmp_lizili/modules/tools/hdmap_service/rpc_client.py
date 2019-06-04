#! /usr/bin/python

import grpc

import modules.msgs.hdmap.proto.rpc_service_pb2 as rpc_service_pb2
import modules.msgs.hdmap.proto.rpc_service_pb2_grpc as rpc_service_pb2_grpc
from modules.common.proto.geometry_pb2 import PointENU

RPC_CHANNEL = "localhost:9999"


class RpcClient:
    def __init__(self, channel=RPC_CHANNEL, keepalive_timeout_ms=20000):
        channel = grpc.insecure_channel(target=channel, options=[(
            'grpc.keepalive_timeout_ms', keepalive_timeout_ms)])
        self.stub = rpc_service_pb2_grpc.RpcServiceNodeStub(channel)

    def GetAvailableMaps(self):
        request = rpc_service_pb2.GetAvailableMapsRequest()
        try:
            response = self.stub.GetAvailableMaps(request)
        except grpc.RpcError as e:
            raise e

        return response

    def SetMap(self, map_file):
        request = rpc_service_pb2.SetMapRequest(map=map_file)
        try:
            response = self.stub.SetMap(request)
        except grpc.RpcError as e:
            raise e

        return response

    def SetRoute(self, route_name):
        request = rpc_service_pb2.SetRouteRequest()
        request.route_name = route_name
        try:
            response = self.stub.SetRoute(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetCurrMap(self):
        request = rpc_service_pb2.GetCurrMapRequest()
        try:
            response = self.stub.GetCurrMap(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetCurrRoute(self):
        request = rpc_service_pb2.GetCurrRouteRequest()
        try:
            response = self.stub.GetCurrRoute(request)
        except grpc.RpcError as e:
            raise e

        return response

    def SetRoutingPoints(self, start, end):
        request = rpc_service_pb2.SetRoutingPointsRequest()
        request.points.extend([start, end])
        try:
            response = self.stub.SetRoutingPoints(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetLocalMap(self, location, radius=200):
        request = rpc_service_pb2.GetLocalMapRequest(
            location=location, radius=radius)
        try:
            response = self.stub.GetLocalMap(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetLocalPath(self,
                     location,
                     forward_distance=100,
                     backward_distance=100):
        request = rpc_service_pb2.GetLocalPathRequest(
            location=location,
            forward_distance=forward_distance,
            backward_distance=backward_distance)
        try:
            response = self.stub.GetLocalPath(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetMapElements(self, map_units):
        request = rpc_service_pb2.GetMapElementsRequest(map_units=map_units)
        try:
            response = self.stub.GetMapElements(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetPointOnRoad(self, location, radius, points):
        request = rpc_service_pb2.GetPointOnRoadRequest(
            location=location, radius=radius, points=points)
        try:
            response = self.stub.GetPointOnRoad(request)
        except grpc.RpcError as e:
            raise e

        return response

    def GetMapWarnings(self):
        request = rpc_service_pb2.GetMapWarningsRequest()
        try:
            response = self.stub.GetMapWarnings(request)
        except grpc.RpcError as e:
            raise e

        return response
