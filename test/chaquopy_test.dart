import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import '../lib/chaquopy.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  // MethodChannel channel = const MethodChannel('chaquopy');

  // channel.setMockMethodCallHandler(null);

  group('test', () {
    test('Check standard code', () async {
      expect(await Chaquopy.executeCode('print("Hello")'),
          {"textOutput": "Hello", "error": ""});
    });
    test('Check standard code', () async {
      expect(await Chaquopy.executeFunction('return 2'),
          {"textOutput": "Hello", "error": ""});
    });
  });
}
