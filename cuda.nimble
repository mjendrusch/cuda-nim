mode = ScriptMode.Verbose

packageName    = "cuda"
version        = "0.1.0"
author         = "Michael Jendrusch"
description    = "High level bindings to the CUDA runtime and runtime compilation."
license        = "MIT"
skipDirs       = @["tests", "examples"]
skipFiles      = @["cuda.html", "api.html"]

requires "nim >= 0.17.0"

--forceBuild

proc testConfig() =
  --hints: off
  --linedir: on
  --stacktrace: on
  --linetrace: on
  --debuginfo
  --path: "."
  --run

proc exampleConfig() =
  --define: release
  --path: "."

task test, "run cuda tests":
  testConfig()
  setCommand "c", "tests/tall.nim"

task examples, "build cuda example applications":
  exampleConfig()
