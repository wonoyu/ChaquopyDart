package com.chaquopy.chaquopy

import androidx.annotation.NonNull
import android.util.Log
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.util.*

// JSON parsing
import org.json.JSONObject

/** ChaquopyPlugin */
class ChaquopyPlugin : FlutterPlugin, MethodCallHandler {
    /// The MethodChannel that will the communication between Flutter and native Android
    ///
    /// This local reference serves to register the plugin with the Flutter Engine and unregister it
    /// when the Flutter Engine is detached from the Activity
    private lateinit var channel: MethodChannel

    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "chaquopy")
        channel.setMethodCallHandler(this)
    }

    //  * This will run python code consisting of error and result output...
    fun _runPythonTextCode(code: String): Map<String, Any?> {
        val _returnOutput: MutableMap<String, Any?> = HashMap()
        val _python: Python = Python.getInstance()
        val _console: PyObject = _python.getModule("script")
        val _sys: PyObject = _python.getModule("sys")
        val _io: PyObject = _python.getModule("io")

        return try {
            val _textOutputStream: PyObject = _io.callAttr("StringIO")
            _sys["stdout"] = _textOutputStream
            _console.callAttrThrows("mainTextCode", code)
            _returnOutput["textOutputOrError"] = _textOutputStream.callAttr("getvalue").toString()
            _returnOutput
        } catch (e: PyException) {
            _returnOutput["textOutputOrError"] = e.message.toString()
            _returnOutput
        }
    }

    //  * This will build and run a python function, returning an error and the result of said function
    fun _runPythonTextFunction(name: String, code: String, args: Array<String>): Map<String, Any?> {
        val _returnOutput: MutableMap<String, Any?> = HashMap()
        val _returns: Any?;
        val _python: Python = Python.getInstance()
        val _build_n_run: PyObject = _python.getModule("build_n_run")
        val _sys: PyObject = _python.getModule("sys")
        val _io: PyObject = _python.getModule("io")

        return try {
            val _textOutputStream: PyObject = _io.callAttr("StringIO")
            _returns = _build_n_run.callAttrThrows("mainTextCode", name, code, args.map { it.toString() }.toTypedArray(), _textOutputStream)
            _returnOutput["textOutputOrError"] = _returns.toString() // _textOutputStream.callAttr("getvalue").toString()
            _returnOutput
        } catch (e: PyException) {
            _returnOutput["textOutputOrError"] = "PyException => " + e.message.toString()
            _returnOutput
        }
    }

    fun _runPreprocess(img: ByteArray): ByteArray {
        val _py: Python = Python.getInstance()
        val _module: PyObject = _py.getModule("preprocess")

        return try {
            val preprocessed = _module.callAttr("preprocess", img).toJava(ByteArray::class.java)
            Log.d("Preprocessed", preprocessed.toString())
            preprocessed
        } catch (e: PyException) {
            Log.d("PyException/PreprocessFailed", e.message.toString())
            img
        }
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "runPythonScript" -> {
                try {
                val code: String = call.arguments() ?: ""
                val _result: Map<String, Any?> = _runPythonTextCode(code)
                result.success(_result)
            } catch (e: Exception) {
                val _result: MutableMap<String, Any?> = HashMap()
                _result["textOutputOrError"] = e.message.toString()
                result.success(_result)
            }
            }
            "runPythonFunction" -> {
                try {
                val name: String = call.argument("name") ?: ""
                val code: String = call.argument("code") ?: ""
                val _args: ArrayList<String> = call.argument("args") ?: arrayListOf<String>()
                val args: Array<String> = _args.toTypedArray() ?: arrayOf<String>()
                val _result: Map<String, Any?> = _runPythonTextFunction(name, code, args)
                result.success(_result)
            } catch (e: Exception) {
                val _result: MutableMap<String, Any?> = HashMap()
                println(call.argument("args"))
                _result["textOutputOrError"] = "Exception during method call => " + e.message.toString() + e.getStackTrace()[0].getLineNumber().toString()
                result.success(_result)
            }
            }
            "runPreprocess" -> {
                try {
                    val _img: ByteArray = call.argument("img")!!
                    val _res: ByteArray = _runPreprocess(_img)
                    result.success(_res)
                } catch (e: Exception) {
                    Log.d("Exception", e.message.toString())
                    result.success(call.argument("img")!!)
                }
            }
        }
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }
}
