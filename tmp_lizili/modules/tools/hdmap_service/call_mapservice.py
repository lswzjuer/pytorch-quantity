#! /usr/bin/python

import argparse
import grpc
from rpc_client import RpcClient
from modules.common.proto.geometry_pb2 import PointENU
from modules.msgs.hdmap.proto.hdmap_common_pb2 import MapUnit

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test for GetLocalMap service.")
    parser.add_argument(
        "service_name", action="store", type=str, help="the input test case.")
    parser.add_argument(
        "--map", type=str, default="hangzhou", help="the map name.")
    parser.add_argument(
        "--route", type=str, default="route", help="the route name.")
    parser.add_argument("--x", type=float, default=0, help="the x coordinate.")
    parser.add_argument("--y", type=float, default=0, help="the y coordinate.")
    parser.add_argument("--start_x", type=float, default=0,
                        help="the start x coordinate of route.")
    parser.add_argument("--start_y", type=float, default=0,
                        help="the start y coordinate of route.")
    parser.add_argument("--end_x", type=float, default=0,
                        help="the end x coordinate of route.")
    parser.add_argument("--end_y", type=float, default=0,
                        help="the end y coordinate of route.")
    parser.add_argument(
        "--forward", type=int, default=100, help="the forward distance.")
    parser.add_argument(
        "--backward", type=int, default=100, help="the backward distance.")
    parser.add_argument("--radius", type=float,
                        default=100, help="the radius.")
    parser.add_argument("--id", type=int, default=0, help="the section id.")
    parser.add_argument("--host", type=str,
                        default="localhost", help="the rpc host.")
    parser.add_argument("--port", type=str, default="9999",
                        help="the rpc port.")

    args = parser.parse_args()
    rpc_channel = args.host + ":" + args.port
    try:
        client = RpcClient(channel=rpc_channel)
    except:
        print("Connect Hdmap Service Failed")

    if args.service_name == "GetAvailableMaps":
        print("GetAvailableMaps")
        print(client.GetAvailableMaps())
    elif args.service_name == "GetLocalPath":
        print("GetLocalPath at ({0}, {1}) forward {2} and backward {3}".format(
            args.x, args.y, args.forward, args.backward))
        print(client.GetLocalPath(
            PointENU(x=args.x, y=args.y), args.forward, args.backward))
    elif args.service_name == "SetMap":
        print("Set map to: ", args.map)
        print(client.SetMap(args.map))
    elif args.service_name == "SetRoute":
        print("Set route to: ", args.route)
        print(client.SetRoute(args.route))
    elif args.service_name == "SetRoutingPoints":
        print(client.SetRoutingPoints(
            PointENU(x=args.start_x, y=args.start_y),
            PointENU(x=args.end_x, y=args.end_y)))
    elif args.service_name == "GetLocalMap":
        print("GetLocalMap at ({0}, {1}) radius {2}".format(
            args.x, args.y, args.radius))
        print(client.GetLocalMap(PointENU(x=args.x, y=args.y), args.radius))
    elif args.service_name == "GetSection":
        print("GetSection: ", args.id)
        map_units = []
        section_unit = MapUnit()
        section_unit.id = args.id
        section_unit.type = MapUnit.MAP_UNIT_SECTION
        map_units.append(section_unit)
        print(client.GetMapElements(map_units))
    elif args.service_name == "GetConnection":
        print("GetConnection: ", args.id)
        map_units = []
        conn_unit = MapUnit()
        conn_unit.id = args.id
        conn_unit.type = MapUnit.MAP_UNIT_CONNECTION
        map_units.append(conn_unit)
        print(client.GetMapElements(map_units))
    elif args.service_name == "GetMap":
        print("GetMap")
        try:
            response = client.GetCurrMap()
            print(response.map_name)
        except grpc.RpcError as e:
            print("")
    elif args.service_name == "GetRoute":
        print("GetRoute")
        try:
            response = client.GetCurrRoute()
            print(response.route_name)
        except grpc.RpcError as e:
            print("")
    elif args.service_name == "GetPointOnRoad":
        print("GetPointOnRoad")
        print("GetPointOnRoad at ({0}, {1})".format(args.x, args.y))
        print(client.GetPointOnRoad(
            PointENU(x=args.x, y=args.y), args.radius, [PointENU(x=args.x, y=args.y)]))
    else:
        print("Unimplemented service")
