const libRtc = "nvrtc.so"

type
  NvrtcResult* {.size: sizeof(cint).} = enum
    NVRTC_SUCCESS = 0, NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2, NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4, NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6, NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10, NVRTC_ERROR_INTERNAL_ERROR = 11
  NvrtcProgram* = ptr object

proc nvrtcGetErrorString*(result: NvrtcResult): cstring {.
    importc: "nvrtcGetErrorString", dynlib: libRtc.}

proc nvrtcVersion*(major: ptr cint; minor: ptr cint): NvrtcResult {.
    importc: "nvrtcVersion", dynlib: libRtc.}

proc nvrtcCreateProgram*(prog: ptr NvrtcProgram; src: cstring; name: cstring;
                        numHeaders: cint; headers: cstringArray;
                        includeNames: cstringArray): NvrtcResult {.
    importc: "nvrtcCreateProgram", dynlib: libRtc.}

proc nvrtcDestroyProgram*(prog: ptr NvrtcProgram): NvrtcResult {.
    importc: "nvrtcDestroyProgram", dynlib: libRtc.}

proc nvrtcCompileProgram*(prog: NvrtcProgram; numOptions: cint; options: cstringArray): NvrtcResult {.
    importc: "nvrtcCompileProgram", dynlib: libRtc.}

proc nvrtcGetPTXSize*(prog: NvrtcProgram; ptxSizeRet: ptr csize): NvrtcResult {.
    importc: "nvrtcGetPTXSize", dynlib: libRtc.}

proc nvrtcGetPTX*(prog: NvrtcProgram; ptx: cstring): NvrtcResult {.
    importc: "nvrtcGetPTX", dynlib: libRtc.}

proc nvrtcGetProgramLogSize*(prog: NvrtcProgram; logSizeRet: ptr csize): NvrtcResult {.
    importc: "nvrtcGetProgramLogSize", dynlib: libRtc.}

proc nvrtcGetProgramLog*(prog: NvrtcProgram; log: cstring): NvrtcResult {.
    importc: "nvrtcGetProgramLog", dynlib: libRtc.}

proc nvrtcAddNameExpression*(prog: NvrtcProgram; nameExpression: cstring): NvrtcResult {.
    importc: "nvrtcAddNameExpression", dynlib: libRtc.}

proc nvrtcGetLoweredName*(prog: NvrtcProgram; nameExpression: cstring;
                         loweredName: cstringArray): NvrtcResult {.
    importc: "nvrtcGetLoweredName", dynlib: libRtc.}
