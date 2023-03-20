import 'dart:async';

import 'package:flutter/services.dart';

/// static class for accessing the executeCode function.
class Chaquopy {
  static const MethodChannel _channel = const MethodChannel('chaquopy');

  /// This function execute your python code and returns result Map.
  /// Structure of result map is :
  /// result['textOutput'] : The original output / error
  static Future<Map<String, dynamic>> executeCode(String code) async {
    dynamic outputData = await _channel.invokeMethod('runPythonScript', code);
    return Map<String, dynamic>.from(outputData);
  }

  static Future<Map<String, dynamic>> executeFunction(
      String name, String code, List<dynamic> args) async {
    dynamic outputData = await _channel.invokeMethod(
        'runPythonFunction', {"name": name, "code": code, "args": args});
    return Map<String, dynamic>.from(outputData);
  }

  static Future<Uint8List> executePreprocess(Uint8List img) async {
    dynamic output = await _channel.invokeMethod('runPreprocess', {"img": img});
    if (output is Uint8List) return output;
    return img;
  }
}
