import cudart, nvrtc

type
  CudaError* = object of Exception
  CompilationError* = object of Exception
  DevicePtr*[T] = distinct CuDevicePtr
  DevicePointer* = CuDevicePtr | DevicePtr
  Dim* = object
    x*, y*, z*: uint

proc cfree*(p: pointer) {. importc: "free" .}

proc errorString(err: CuResult): string =
  var p = cast[ptr cstring](alloc0(sizeOf(pointer)))
  discard err.cuGetErrorString(p)
  result = $p[]
  dealloc p

proc errorString(err: NvRtcResult): string =
  var p = err.nvrtcGetErrorString
  result = $p

template handleError(err: CuResult) =
  let e = err
  if e != CudaSuccess:
    raise newException(CudaError, e.errorString)

template handleErrorRtc(err: NvrtcResult) =
  let e = err
  if e != NvRtcSuccess:
    raise newException(CompilationError, e.errorString)

proc getDeviceCount*: int =
  var res: cint
  handleError cuDeviceGetCount(res.addr)
  res

proc init* =
  handleError cuInit(0)

proc getDevice*(ordinal: int): CuDevice =
  handleError cuDeviceGet(result.addr, ordinal.cint)

proc name*(device: CuDevice): string =
  ## Get the name of a CUDA device, limited to 128 characters.
  var cs = cast[cstring](alloc0(128 * sizeOf(char)))
  handleError cuDeviceGetName(cs, 128, device)
  result = $cs
  dealloc cs

proc computeCapability*(device: CuDevice): tuple[major, minor: cint] =
  ## Get the major and minor compute capability of a device.
  handleError cuDeviceComputeCapability(result.major.addr,
                                        result.minor.addr,
                                        device)

proc totalMem*(device: CuDevice): csize =
  ## Get the number of bytes of memory available on a CUDA davice.
  handleError cuDeviceTotalMem(result.addr, device)

## Program procedures
proc newProgram*(src: string; name: string = "defaultName.cu";
                 numHeaders: int = 0; headers: seq[string] = @[];
                 includeNames: seq[string] = @[]): NvrtcProgram =
  ## Creates a new CUDA program from source.
  var
    headersPtr: cstringArray = nil
    includeNamesPtr: cstringArray = nil
  if numHeaders != 0:
    headersPtr = allocCstringArray(headers)
    includeNamesPtr = allocCstringArray(includeNames)

  handleErrorRtc nvrtcCreateProgram(result.addr, src.cstring, name.cstring,
                                    headers.len.cint, headersPtr,
                                    includeNamesPtr)
  if numHeaders != 0:
    deallocCstringArray(headersPtr)
    deallocCstringArray(includeNamesPtr)

proc compile*(pr: NvrtcProgram; options: seq[string] = @[]) =
  ## Compiles a given CUDA program, setting options.
  var optionsPtr: cstringArray = nil
  if options.len != 0:
    optionsPtr = allocCstringArray(options)
  handleErrorRtc nvrtcCompileProgram(pr, options.len.cint, optionsPtr)
  if options.len != 0:
    deallocCstringArray(optionsPtr)

proc ptxSize*(pr: NvrtcProgram): uint =
  ## Computes the size of the PTX code in bytes resulting from a program.
  var res: csize
  handleErrorRtc nvrtcGetPtxSize(pr, res.addr)
  result = res.uint

proc ptx*(pr: NvrtcProgram): string =
  ## Generates PTX from a program.
  let cs = cast[cstring](alloc0(pr.ptxSize.int * sizeOf(char)))
  defer: dealloc(cs)
  handleErrorRtc nvrtcGetPtx(pr, cs)
  result = $cs

proc dispose*(pr: NvrtcProgram) =
  ## Destroys an NvrtcProgram.
  handleErrorRtc pr.unsafeAddr.nvrtcDestroyProgram

## Context procedures
proc detach*(ctx: CuContext) =
  ## Decrease the refcount of a CuContext.
  handleError cuCtxDetach(ctx)

proc newContext*(device: CuDevice; flags: CuCtxFlags = {0.CuCtxFlag}): CuContext =
  ## Create a new CUDA context on top of a CuDevice.
  try:
    handleError cuCtxCreate(result.addr, cast[cuint](flags), device)
  except CudaError as ce:
    result.detach
    raise ce

proc version*(ctx: CuContext): uint =
  ## Get the version of the CUDA API for ctx.
  var cu: cuint
  handleError cuCtxGetApiVersion(ctx, cu.addr)
  result = cu

proc currentFlags*: CuCtxFlags =
  ## Get all flags set for the current context.
  handleError cuCtxGetFlags(cast[ptr cuint](result.addr))

proc pushCurrent*(ctx: CuContext) =
  ## Push ctx onto the CPU thread.
  handleError cuCtxPushCurrent(ctx)

proc popCurrent*: CuContext =
  ## Pops the current context from the CPU thread.
  handleError cuCtxPopCurrent(result.addr)

proc setCurrent*(ctx: CuContext) =
  ## Sets ctx as the current context for the CPU thread.
  handleError cuCtxSetCurrent(ctx)

proc synchCurrent* =
  ## Synchronizes all tasks in the current context.
  handleError cuCtxSynchronize()

## Module procedures
proc newModule*(path: string): CuModule =
  ## Load a CUDA module from a path.
  let cs = path.cstring
  handleError cuModuleLoad(result.addr, cs)

proc newModuleFromData*(data: string): CuModule =
  ## Load a CUDA module from a PTX buffer.
  let cs = data.cstring
  handleError cuModuleLoadDataEx(result.addr, cs, 0.cuint, nil, nil)

proc getKernel*(md: CuModule; name: string): CuFunction =
  ## Get a CUDA kernel from a module by name.
  let cs = name.cstring
  handleError cuModuleGetFunction(result.addr, md, cs)

proc getGlobal*(md: CuModule; name: string): tuple[p: CuDevicePtr, len: int] =
  ## Get a reference to a global variable by name.
  let
    cs = name.cstring
  var
    size: csize
  handleError cuModuleGetGlobal(result.p.addr, size.addr, md, cs)
  result.len = size.int

proc unload*(md: CuModule) =
  ## Unload a module from the current Context.
  handleError md.cuModuleUnload

## Memory management functions
proc hostAlloc*(bytes: uint): pointer =
  ## Allocates bytes page-locked memory on the host.
  handleError cuMemAllocHost(result.addr, bytes.csize)

proc hostFree*(p: pointer) =
  ## Frees page-locked host memory.
  handleError cuMemFreeHost(p)

proc deviceAlloc*(bytes: uint): CuDevicePtr =
  ## Allocate bytes memory on a device.
  handleError cuMemAlloc(result.addr, bytes.csize)

proc deviceFree*(p: CuDevicePtr) =
  ## Free memory allocated on a device.
  handleError cuMemFree(p)

proc copyMem*(dest: DevicePointer; src: pointer; size: Natural) =
  ## Copy data from the host to the device.
  handleError cuMemcpyHtoD(dest.CuDevicePtr, src, size)

proc copyMem*(dest: pointer; src: DevicePointer; size: Natural) =
  ## Copy data from the device back to the host.
  handleError cuMemcpyDtoH(dest, src.CuDevicePtr, size)

proc copyMem*(dest, src: DevicePointer; size: Natural) =
  ## Copy data from one device address to another.
  handleError cuMemcpyHtoH(dest.CuDevicePtr, src.CuDevicePtr, size)

template gpu*[T](x: ptr T; num: uint): auto =
  ## Copy a pointer to gpu.
  var deviceX = deviceAlloc(num * sizeOf(x[].type).uint)
  copyMem(deviceX, x, num * sizeOf(x[].type).uint)
  DevicePtr[x[].type](deviceX)

template gpu*[T](x: openarray[T]): auto =
  ## Copy an array or seq to gpu.
  x[0].unsafeAddr.gpu x.len.uint

template host*[T](dest: var ptr T; x: CuDevicePtr; num: uint) =
  ## Copy a device pointer to the host
  dest.realloc(num * sizeOf(T).uint)
  dest.copyMem(CuDevicePtr x, num * sizeOf(T).uint)

template host*[T](dest: var seq[T]; x: CuDevicePtr; num: uint) =
  ## Copy a device pointer to the host
  dest.setLen(num)
  dest[0].addr.copyMem(CuDevicePtr x, num * sizeOf(T).uint)

proc setMem*[T](dest: DevicePointer; val: T; size: Natural) =
  ## Sets size values at dest to val.
  case sizeOf T
  of 1:
    handleError cuMemsetD8(dest, val.cuchar, size)
  of 2:
    handleError cuMemsetD16(dest, val.cushort, size)
  of 4:
    handleError cuMemsetD32(dest, val.cushort, size)
  else:
    raise newException(CudaError, "Illegal bitwidth for cuMemset!")

## Kernel execution
proc launch*(f: CuFunction; gridDimX, gridDimY, gridDimZ,
             blockDimX, blockDimY, blockDimZ: Natural;
             params: openarray[pointer]; sharedBytes: Natural = 0) =
  ## Launch the kernel associated with f.
  handleError cuLaunchKernel(f, gridDimX.cuint, gridDimY.cuint, gridDimZ.cuint,
                             blockDimX.cuint, blockDimY.cuint, blockDimZ.cuint,
                             sharedBytes.cuint, nil, params[0].unsafeAddr, nil)

proc launch*(f: CuFunction; gridDim, blockDim: Dim;
             params: openarray[pointer]; sharedBytes: Natural = 0) =
  ## Launch the kernel associated with f using a simplified interface.
  f.launch(gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
           params, sharedBytes)

## Autoinit
template initCuda*(at: int) =
  bind init
  init()
  var
    baseDevice {. inject .}: CuDevice
    baseContext {. inject .}: CuContext
  try:
    baseDevice = getDevice(at)
    baseContext = newContext(baseDevice)
  except:
    echo "Device at " & $at & " not accessible. Using Device 0 instead."
    baseDevice = getDevice(0)
    baseContext = newContext(baseDevice)
