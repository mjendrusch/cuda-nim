import cuda

type
  CudaError* = object of Exception

proc cfree*(p: pointer) {. importc: "free" .}

proc errorString(err: CuResult): string =
  var p: ptr cstring
  discard err.cuGetErrorString(p)
  result = $p[]
  p.cfree

template handleError(err: CuResult) =
  let e = err
  if e != CudaSuccess:
    raise newException(CudaError, e.errorString)

proc getDeviceCount*: int =
  var res: cint
  handleError cuDeviceGetCount(res.addr)
  res

proc getDevice*(ordinal: int): CuDevice =
  handleError cuDeviceGet(result.addr, ordinal.cint)
