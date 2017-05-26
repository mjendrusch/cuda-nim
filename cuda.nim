const
  libCuda = "cuda.so"

const
  CUDA_API_VERSION* = 8000

const
  CUDA_VERSION* = 8000


type
  CuDevicePtr* = distinct uint

type
  CuDevice* = cint
  CuCtxSt* {. incompleteStruct .} = object
  CuModSt* {. incompleteStruct .} = object
  CuFuncSt* {. incompleteStruct .} = object
  CuArraySt* {. incompleteStruct .} = object
  CuMipMappedArraySt* {. incompleteStruct .} = object
  CuTexRefSt* {. incompleteStruct .} = object
  CuSurfRefSt* {. incompleteStruct .} = object
  CuEventSt* {. incompleteStruct .} = object
  CuStreamSt* {. incompleteStruct .} = object
  CuGraphicsResourceSt* {. incompleteStruct .} = object
  CuLinkStateSt* {. incompleteStruct .} = object
  CUcontext* = ptr CUctxSt
  CUmodule* = ptr CUmodSt
  CUfunction* = ptr CUfuncSt
  CUarray* = ptr CUarraySt
  CUmipmappedArray* = ptr CUmipmappedArraySt
  CUtexref* = ptr CUtexrefSt
  CUsurfref* = ptr CUsurfrefSt
  CUevent* = ptr CUeventSt
  CUstream* = ptr CUstreamSt
  CUgraphicsResource* = ptr CUgraphicsResourceSt
  CUtexObject* = culonglong
  CUsurfObject* = culonglong
  CUuuid* = object
    bytes*: array[16, char]


when Cuda_Api_Version >= 4010:
  const
    CU_IPC_HANDLE_SIZE* = 64
  type
    CUipcEventHandle* = object
      reserved*: array[Cu_Ipc_Handle_Size, char]
    CUipcMemHandle* = object
      reserved*: array[Cu_Ipc_Handle_Size, char]
    CUipcMemFlags* {.size: sizeof(cint).} = enum
      CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x00000001

type
  CUmemAttachFlags* {.size: sizeof(cint).} = enum
    CU_MEM_ATTACH_GLOBAL = 0x00000001, CU_MEM_ATTACH_HOST = 0x00000002,
    CU_MEM_ATTACH_SINGLE = 0x00000004
  CuCtxFlag* {.size: sizeof(cint).} = enum
    CU_CTX_SCHED_AUTO = 0x00000000, CU_CTX_SCHED_SPIN = 0x00000001,
    CU_CTX_SCHED_YIELD = 0x00000002, CU_CTX_SCHED_BLOCKING_SYNC = 0x00000004,
    CU_CTX_SCHED_MASK = 0x00000007, CU_CTX_MAP_HOST = 0x00000008,
    CU_CTX_LMEM_RESIZE_TO_MAX = 0x00000010, CU_CTX_FLAGS_MASK = 0x0000001F
  CuCtxFlags* = set[CuCtxFlag]
  CUstreamFlags* {.size: sizeof(cint).} = enum
    CU_STREAM_DEFAULT = 0x00000000, CU_STREAM_NON_BLOCKING = 0x00000001
  CUeventFlags* {.size: sizeof(cint).} = enum
    CU_EVENT_DEFAULT = 0x00000000, CU_EVENT_BLOCKING_SYNC = 0x00000001,
    CU_EVENT_DISABLE_TIMING = 0x00000002, CU_EVENT_INTERPROCESS = 0x00000004

const
  CU_STREAM_LEGACY* = (cast[CUstream](0x00000001))
  CU_STREAM_PER_THREAD* = (cast[CUstream](0x00000002))
  CU_CTX_BLOCKING_SYNC* = CU_CTX_SCHED_BLOCKING_SYNC



when Cuda_Api_Version >= 8000:
  type
    CUstreamWaitValueFlags* {.size: sizeof(cint).} = enum
      CU_STREAM_WAIT_VALUE_GEQ = 0x00000000, CU_STREAM_WAIT_VALUE_EQ = 0x00000001,
      CU_STREAM_WAIT_VALUE_AND = 0x00000002, CU_STREAM_WAIT_VALUE_FLUSH = 1 shl 30
    CUstreamWriteValueFlags* {.size: sizeof(cint).} = enum
      CU_STREAM_WRITE_VALUE_DEFAULT = 0x00000000,
      CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x00000001
    CUstreamBatchMemOpType* {.size: sizeof(cint).} = enum
      CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1, CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
      CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
    INNER_C_UNION_1190273278* = object {.union.}
      value*: uint32
      pad*: uint64
    CUstreamMemOpWaitValueParams_st_2363022978* = object
      operation*: CUstreamBatchMemOpType
      address*: CUdeviceptr
      ano2751382075*: INNER_C_UNION_1190273278
      flags*: cuint
      alias*: CUdeviceptr
    INNER_C_UNION_2329837567* = object {.union.}
      value*: uint32
      pad*: uint64
    CUstreamMemOpWriteValueParams_st_3502587268* = object
      operation*: CUstreamBatchMemOpType
      address*: CUdeviceptr
      ano3890946364*: INNER_C_UNION_2329837567
      flags*: cuint
      alias*: CUdeviceptr
    CUstreamMemOpFlushRemoteWritesParams_st_347249857* = object
      operation*: CUstreamBatchMemOpType
      flags*: cuint
    CUstreamBatchMemOpParams* = object {.union.}
      operation*: CUstreamBatchMemOpType
      waitValue*: CUstreamMemOpWaitValueParams_st_2363022978
      writeValue*: CUstreamMemOpWriteValueParams_st_3502587268
      flushRemoteWrites*: CUstreamMemOpFlushRemoteWritesParams_st_347249857
      pad*: array[6, uint64]


type
  CUoccupancyFlags* {.size: sizeof(cint).} = enum
    CU_OCCUPANCY_DEFAULT = 0x00000000,
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x00000001
  CUarrayFormat* {.size: sizeof(cint).} = enum
    CU_AD_FORMAT_UNSIGNED_INT8 = 0x00000001,
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x00000002,
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x00000003,
    CU_AD_FORMAT_SIGNED_INT8 = 0x00000008, CU_AD_FORMAT_SIGNED_INT16 = 0x00000009,
    CU_AD_FORMAT_SIGNED_INT32 = 0x0000000A, CU_AD_FORMAT_HALF = 0x00000010,
    CU_AD_FORMAT_FLOAT = 0x00000020
  CUaddressMode* {.size: sizeof(cint).} = enum
    CU_TR_ADDRESS_MODE_WRAP = 0, CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2, CU_TR_ADDRESS_MODE_BORDER = 3
  CUfilterMode* {.size: sizeof(cint).} = enum
    CU_TR_FILTER_MODE_POINT = 0, CU_TR_FILTER_MODE_LINEAR = 1
  CUdeviceAttribute* {.size: sizeof(cint).} = enum
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10, CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34, CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_MAX
  CUdevprop* = object
    maxThreadsPerBlock*: cint
    maxThreadsDim*: array[3, cint]
    maxGridSize*: array[3, cint]
    sharedMemPerBlock*: cint
    totalConstantMemory*: cint
    sIMDWidth*: cint
    memPitch*: cint
    regsPerBlock*: cint
    clockRate*: cint
    textureAlign*: cint
  CUpointerAttribute* {.size: sizeof(cint).} = enum
    CU_POINTER_ATTRIBUTE_CONTEXT = 1, CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3, CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7, CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
  CUfunctionAttribute* {.size: sizeof(cint).} = enum
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4, CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX
  CUfuncCache* {.size: sizeof(cint).} = enum
    CU_FUNC_CACHE_PREFER_NONE = 0x00000000,
    CU_FUNC_CACHE_PREFER_SHARED = 0x00000001, CU_FUNC_CACHE_PREFER_L1 = 0x00000002,
    CU_FUNC_CACHE_PREFER_EQUAL = 0x00000003
  CUsharedconfig* {.size: sizeof(cint).} = enum
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00000000,
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x00000001,
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x00000002
  CUmemorytype* {.size: sizeof(cint).} = enum
    CU_MEMORYTYPE_HOST = 0x00000001, CU_MEMORYTYPE_DEVICE = 0x00000002,
    CU_MEMORYTYPE_ARRAY = 0x00000003, CU_MEMORYTYPE_UNIFIED = 0x00000004
  CUcomputemode* {.size: sizeof(cint).} = enum
    CU_COMPUTEMODE_DEFAULT = 0, CU_COMPUTEMODE_PROHIBITED = 2,
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
  CUmemAdvise* {.size: sizeof(cint).} = enum
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1, CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4, CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6
  CUmemRangeAttribute* {.size: sizeof(cint).} = enum
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4
  CUjitOption* {.size: sizeof(cint).} = enum
    cjoMAX_REGISTERS = 0, cjoTHREADS_PER_BLOCK, cjoWALL_TIME,
    cjoINFO_LOG_BUFFER, cjoINFO_LOG_BUFFER_SIZE_BYTES,
    cjoERROR_LOG_BUFFER, cjoERROR_LOG_BUFFER_SIZE_BYTES,
    cjoOPTIMIZATION_LEVEL, cjoTARGET_FROM_CUCONTEXT, cjoTARGET,
    cjoFALLBACK_STRATEGY, cjoGENERATE_DEBUG_INFO, cjoLOG_VERBOSE,
    cjoGENERATE_LINE_INFO, cjoCACHE_MODE, cjoNEW_SM3X_OPT,
    cjoFAST_COMPILE, cjoNUM_OPTIONS
  CUjitTarget* {.size: sizeof(cint).} = enum
    cjtCOMPUTE_10 = 10, cjtCOMPUTE_11 = 11, cjtCOMPUTE_12 = 12,
    cjtCOMPUTE_13 = 13, cjtCOMPUTE_20 = 20, cjtCOMPUTE_21 = 21,
    cjtCOMPUTE_30 = 30, cjtCOMPUTE_32 = 32, cjtCOMPUTE_35 = 35,
    cjtCOMPUTE_37 = 37, cjtCOMPUTE_50 = 50, cjtCOMPUTE_52 = 52,
    cjtCOMPUTE_53 = 53, cjtCOMPUTE_60 = 60, cjtCOMPUTE_61 = 61,
    cjtCOMPUTE_62 = 62
  CUjitFallback* {.size: sizeof(cint).} = enum
    CU_PREFER_PTX = 0, CU_PREFER_BINARY
  CUjitCacheMode* {.size: sizeof(cint).} = enum
    cjcNONE = 0, cjcCG, cjcCA
  CUjitInputType* {.size: sizeof(cint).} = enum
    cjiCUBIN = 0, cjiPTX, cjiFATBINARY,
    cjiOBJECT, cjiLIBRARY, cjiNUMTYPES

const
  CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK* = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
  CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK* = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH* = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT* = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES* = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS

type
  CUlinkState* = ptr CUlinkStateSt
  CUgraphicsRegisterFlags* {.size: sizeof(cint).} = enum
    CU_GRAPHICS_REGISTER_FLAGS_NONE = 0x00000000,
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 0x00000001,
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x00000002,
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x00000004,
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x00000008

  CUgraphicsMapResourceFlags* {.size: sizeof(cint).} = enum
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00000000,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x00000001,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x00000002

  CUarrayCubemapFace* {.size: sizeof(cint).} = enum
    CU_CUBEMAP_FACE_POSITIVE_X = 0x00000000,
    CU_CUBEMAP_FACE_NEGATIVE_X = 0x00000001,
    CU_CUBEMAP_FACE_POSITIVE_Y = 0x00000002,
    CU_CUBEMAP_FACE_NEGATIVE_Y = 0x00000003,
    CU_CUBEMAP_FACE_POSITIVE_Z = 0x00000004,
    CU_CUBEMAP_FACE_NEGATIVE_Z = 0x00000005

  CUlimit* {.size: sizeof(cint).} = enum
    CU_LIMIT_STACK_SIZE = 0x00000000, CU_LIMIT_PRINTF_FIFO_SIZE = 0x00000001,
    CU_LIMIT_MALLOC_HEAP_SIZE = 0x00000002,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x00000003,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x00000004, CU_LIMIT_MAX

  CUresourcetype* {.size: sizeof(cint).} = enum
    CU_RESOURCE_TYPE_ARRAY = 0x00000000,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x00000001,
    CU_RESOURCE_TYPE_LINEAR = 0x00000002, CU_RESOURCE_TYPE_PITCH2D = 0x00000003

  CUresult* {.size: sizeof(cint).} = enum
    CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3, CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5, CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8, CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101, CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201, CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205, CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207, CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209, CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211, CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213, CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215, CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217, CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220, CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303, CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400, CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600, CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701, CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708, CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710, CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714, CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716, CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718, CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_PERMITTED = 800, CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999

  CUdeviceP2PAttribute* {.size: sizeof(cint).} = enum
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x00000001,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x00000002,
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x00000003

  CUstreamCallback* = proc (hStream: CUstream; status: CUresult; userData: pointer) {.
      cdecl.}
  CUoccupancyB2DSize* = proc (blockSize: cint): csize {.cdecl.}


const
  CU_MEMHOSTALLOC_PORTABLE* = 0x00000001
  CU_MEMHOSTALLOC_DEVICEMAP* = 0x00000002
  CU_MEMHOSTALLOC_WRITECOMBINED* = 0x00000004
  CU_MEMHOSTREGISTER_PORTABLE* = 0x00000001
  CU_MEMHOSTREGISTER_DEVICEMAP* = 0x00000002
  CU_MEMHOSTREGISTER_IOMEMORY* = 0x00000004

type
  Cuda_Memcpy2d* = object
    srcXInBytes*: csize
    srcY*: csize
    srcMemoryType*: CUmemorytype
    srcHost*: pointer
    srcDevice*: CUdeviceptr
    srcArray*: CUarray
    srcPitch*: csize
    dstXInBytes*: csize
    dstY*: csize
    dstMemoryType*: CUmemorytype
    dstHost*: pointer
    dstDevice*: CUdeviceptr
    dstArray*: CUarray
    dstPitch*: csize
    widthInBytes*: csize
    height*: csize
  Cuda_Memcpy3d* = object
    srcXInBytes*: csize
    srcY*: csize
    srcZ*: csize
    srcLOD*: csize
    srcMemoryType*: CUmemorytype
    srcHost*: pointer
    srcDevice*: CUdeviceptr
    srcArray*: CUarray
    reserved0*: pointer
    srcPitch*: csize
    srcHeight*: csize
    dstXInBytes*: csize
    dstY*: csize
    dstZ*: csize
    dstLOD*: csize
    dstMemoryType*: CUmemorytype
    dstHost*: pointer
    dstDevice*: CUdeviceptr
    dstArray*: CUarray
    reserved1*: pointer
    dstPitch*: csize
    dstHeight*: csize
    widthInBytes*: csize
    height*: csize
    depth*: csize
  Cuda_Memcpy3d_Peer* = object
    srcXInBytes*: csize
    srcY*: csize
    srcZ*: csize
    srcLOD*: csize
    srcMemoryType*: CUmemorytype
    srcHost*: pointer
    srcDevice*: CUdeviceptr
    srcArray*: CUarray
    srcContext*: CUcontext
    srcPitch*: csize
    srcHeight*: csize
    dstXInBytes*: csize
    dstY*: csize
    dstZ*: csize
    dstLOD*: csize
    dstMemoryType*: CUmemorytype
    dstHost*: pointer
    dstDevice*: CUdeviceptr
    dstArray*: CUarray
    dstContext*: CUcontext
    dstPitch*: csize
    dstHeight*: csize
    widthInBytes*: csize
    height*: csize
    depth*: csize
  Cuda_Array_Descriptor* = object
    width*: csize
    height*: csize
    format*: CUarrayFormat
    numChannels*: cuint
  Cuda_Array3d_Descriptor* = object
    width*: csize
    height*: csize
    depth*: csize
    format*: CUarrayFormat
    numChannels*: cuint
    flags*: cuint

  INNER_C_STRUCT_2695263448* = object
    hArray*: CUarray
  INNER_C_STRUCT_742123347* = object
    hMipmappedArray*: CUmipmappedArray
  INNER_C_STRUCT_3083884939* = object
    devPtr*: CUdeviceptr
    format*: CUarrayFormat
    numChannels*: cuint
    sizeInBytes*: csize

  INNER_C_STRUCT_1101100432* = object
    devPtr*: CUdeviceptr
    format*: CUarrayFormat
    numChannels*: cuint
    width*: csize
    height*: csize
    pitchInBytes*: csize
  INNER_C_STRUCT_2540690955* = object
    reserved*: array[32, cint]
  INNER_C_UNION_1914676244* = object {.union.}
    array*: INNER_C_STRUCT_2695263448
    mipmap*: INNER_C_STRUCT_742123347
    linear*: INNER_C_STRUCT_3083884939
    pitch2D*: INNER_C_STRUCT_1101100432
    reserved*: INNER_C_STRUCT_2540690955
  Cuda_Resource_Desc* = object
    resType*: CUresourcetype
    res*: INNER_C_UNION_1914676244
    flags*: cuint
  Cuda_Texture_Desc* = object
    addressMode*: array[3, CUaddressMode]
    filterMode*: CUfilterMode
    flags*: cuint
    maxAnisotropy*: cuint
    mipmapFilterMode*: CUfilterMode
    mipmapLevelBias*: cfloat
    minMipmapLevelClamp*: cfloat
    maxMipmapLevelClamp*: cfloat
    borderColor*: array[4, cfloat]
    reserved*: array[12, cint]
  CUresourceViewFormat* {.size: sizeof(cint).} = enum
    CU_RES_VIEW_FORMAT_NONE = 0x00000000,
    CU_RES_VIEW_FORMAT_UINT_1X8 = 0x00000001,
    CU_RES_VIEW_FORMAT_UINT_2X8 = 0x00000002,
    CU_RES_VIEW_FORMAT_UINT_4X8 = 0x00000003,
    CU_RES_VIEW_FORMAT_SINT_1X8 = 0x00000004,
    CU_RES_VIEW_FORMAT_SINT_2X8 = 0x00000005,
    CU_RES_VIEW_FORMAT_SINT_4X8 = 0x00000006,
    CU_RES_VIEW_FORMAT_UINT_1X16 = 0x00000007,
    CU_RES_VIEW_FORMAT_UINT_2X16 = 0x00000008,
    CU_RES_VIEW_FORMAT_UINT_4X16 = 0x00000009,
    CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0000000A,
    CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0000000B,
    CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0000000C,
    CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0000000D,
    CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0000000E,
    CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0000000F,
    CU_RES_VIEW_FORMAT_SINT_1X32 = 0x00000010,
    CU_RES_VIEW_FORMAT_SINT_2X32 = 0x00000011,
    CU_RES_VIEW_FORMAT_SINT_4X32 = 0x00000012,
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x00000013,
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x00000014,
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x00000015,
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x00000016,
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x00000017,
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x00000018,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x00000019,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x0000001A,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x0000001B,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x0000001C,
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x0000001D,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x0000001E,
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x0000001F,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x00000020,
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x00000021,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x00000022
  Cuda_Resource_View_Desc* = object
    format*: CUresourceViewFormat
    width*: csize
    height*: csize
    depth*: csize
    firstMipmapLevel*: cuint
    lastMipmapLevel*: cuint
    firstLayer*: cuint
    lastLayer*: cuint
    reserved*: array[16, cuint]
  Cuda_Pointer_Attribute_P2p_Tokens* = object
    p2pToken*: culonglong
    vaSpaceToken*: cuint

const
  CUDA_ARRAY3D_LAYERED* = 0x00000001
  CUDA_ARRAY3D_2DARRAY* = 0x00000001
  CUDA_ARRAY3D_SURFACE_LDST* = 0x00000002
  CUDA_ARRAY3D_CUBEMAP* = 0x00000004
  CUDA_ARRAY3D_TEXTURE_GATHER* = 0x00000008
  CUDA_ARRAY3D_DEPTH_TEXTURE* = 0x00000010
  CU_TRSA_OVERRIDE_FORMAT* = 0x00000001
  CU_TRSF_READ_AS_INTEGER* = 0x00000001
  CU_TRSF_NORMALIZED_COORDINATES* = 0x00000002
  CU_TRSF_SRGB* = 0x00000010
  CU_LAUNCH_PARAM_END* = (cast[pointer](0x00000000))
  CU_LAUNCH_PARAM_BUFFER_POINTER* = (cast[pointer](0x00000001))
  CU_LAUNCH_PARAM_BUFFER_SIZE* = (cast[pointer](0x00000002))
  CU_PARAM_TR_DEFAULT* = - 1
  CU_DEVICE_CPU* = -1.CuDevice
  CU_DEVICE_INVALID* = -2.CuDevice



proc cuGetErrorString*(error: CUresult; pStr: ptr cstring): CUresult {.cdecl,
    importc: "cuGetErrorString", dynlib: libcuda.}

proc cuGetErrorName*(error: CUresult; pStr: ptr cstring): CUresult {.cdecl,
    importc: "cuGetErrorName", dynlib: libcuda.}

proc cuInit*(flags: cuint): CUresult {.cdecl, importc: "cuInit", dynlib: libcuda.}

proc cuDriverGetVersion*(driverVersion: ptr cint): CUresult {.cdecl,
    importc: "cuDriverGetVersion", dynlib: libcuda.}

proc cuDeviceGet*(device: ptr CUdevice; ordinal: cint): CUresult {.cdecl,
    importc: "cuDeviceGet", dynlib: libcuda.}

proc cuDeviceGetCount*(count: ptr cint): CUresult {.cdecl,
    importc: "cuDeviceGetCount", dynlib: libcuda.}

proc cuDeviceGetName*(name: cstring; len: cint; dev: CUdevice): CUresult {.cdecl,
    importc: "cuDeviceGetName", dynlib: libcuda.}
proc cuDeviceTotalMem*(bytes: ptr csize; dev: CUdevice): CUresult {.cdecl,
    importc: "cuDeviceTotalMem", dynlib: libcuda.}

proc cuDeviceGetAttribute*(pi: ptr cint; attrib: CUdeviceAttribute; dev: CUdevice): CUresult {.
    cdecl, importc: "cuDeviceGetAttribute", dynlib: libcuda.}

proc cuDeviceGetProperties*(prop: ptr CUdevprop; dev: CUdevice): CUresult {.cdecl,
    importc: "cuDeviceGetProperties", dynlib: libcuda.}

proc cuDeviceComputeCapability*(major: ptr cint; minor: ptr cint; dev: CUdevice): CUresult {.
    cdecl, importc: "cuDeviceComputeCapability", dynlib: libcuda.}

proc cuDevicePrimaryCtxRetain*(pctx: ptr CUcontext; dev: CUdevice): CUresult {.cdecl,
    importc: "cuDevicePrimaryCtxRetain", dynlib: libcuda.}
proc cuDevicePrimaryCtxRelease*(dev: CUdevice): CUresult {.cdecl,
    importc: "cuDevicePrimaryCtxRelease", dynlib: libcuda.}
proc cuDevicePrimaryCtxSetFlags*(dev: CUdevice; flags: cuint): CUresult {.cdecl,
    importc: "cuDevicePrimaryCtxSetFlags", dynlib: libcuda.}
proc cuDevicePrimaryCtxGetState*(dev: CUdevice; flags: ptr cuint; active: ptr cint): CUresult {.
    cdecl, importc: "cuDevicePrimaryCtxGetState", dynlib: libcuda.}
proc cuDevicePrimaryCtxReset*(dev: CUdevice): CUresult {.cdecl,
    importc: "cuDevicePrimaryCtxReset", dynlib: libcuda.}
proc cuCtxCreate*(pctx: ptr CUcontext; flags: cuint; dev: CUdevice): CUresult {.cdecl,
    importc: "cuCtxCreate", dynlib: libcuda.}
proc cuCtxDestroy*(ctx: CUcontext): CUresult {.cdecl, importc: "cuCtxDestroy",
    dynlib: libcuda.}
proc cuCtxPushCurrent*(ctx: CUcontext): CUresult {.cdecl,
    importc: "cuCtxPushCurrent", dynlib: libcuda.}
proc cuCtxPopCurrent*(pctx: ptr CUcontext): CUresult {.cdecl,
    importc: "cuCtxPopCurrent", dynlib: libcuda.}
proc cuCtxSetCurrent*(ctx: CUcontext): CUresult {.cdecl,
    importc: "cuCtxSetCurrent", dynlib: libcuda.}
proc cuCtxGetCurrent*(pctx: ptr CUcontext): CUresult {.cdecl,
    importc: "cuCtxGetCurrent", dynlib: libcuda.}

proc cuCtxGetDevice*(device: ptr CUdevice): CUresult {.cdecl,
    importc: "cuCtxGetDevice", dynlib: libcuda.}
proc cuCtxGetFlags*(flags: ptr cuint): CUresult {.cdecl, importc: "cuCtxGetFlags",
    dynlib: libcuda.}

proc cuCtxSynchronize*(): CUresult {.cdecl, importc: "cuCtxSynchronize",
                                  dynlib: libcuda.}

proc cuCtxSetLimit*(limit: CUlimit; value: csize): CUresult {.cdecl,
    importc: "cuCtxSetLimit", dynlib: libcuda.}

proc cuCtxGetLimit*(pvalue: ptr csize; limit: CUlimit): CUresult {.cdecl,
    importc: "cuCtxGetLimit", dynlib: libcuda.}

proc cuCtxGetCacheConfig*(pconfig: ptr CUfuncCache): CUresult {.cdecl,
    importc: "cuCtxGetCacheConfig", dynlib: libcuda.}

proc cuCtxSetCacheConfig*(config: CUfuncCache): CUresult {.cdecl,
    importc: "cuCtxSetCacheConfig", dynlib: libcuda.}
proc cuCtxGetSharedMemConfig*(pConfig: ptr CUsharedconfig): CUresult {.cdecl,
    importc: "cuCtxGetSharedMemConfig", dynlib: libcuda.}
proc cuCtxSetSharedMemConfig*(config: CUsharedconfig): CUresult {.cdecl,
    importc: "cuCtxSetSharedMemConfig", dynlib: libcuda.}

proc cuCtxGetApiVersion*(ctx: CUcontext; version: ptr cuint): CUresult {.cdecl,
    importc: "cuCtxGetApiVersion", dynlib: libcuda.}

proc cuCtxGetStreamPriorityRange*(leastPriority: ptr cint;
                                 greatestPriority: ptr cint): CUresult {.cdecl,
    importc: "cuCtxGetStreamPriorityRange", dynlib: libcuda.}

proc cuCtxAttach*(pctx: ptr CUcontext; flags: cuint): CUresult {.cdecl,
    importc: "cuCtxAttach", dynlib: libcuda.}

proc cuCtxDetach*(ctx: CUcontext): CUresult {.cdecl, importc: "cuCtxDetach",
    dynlib: libcuda.}

proc cuModuleLoad*(module: ptr CUmodule; fname: cstring): CUresult {.cdecl,
    importc: "cuModuleLoad", dynlib: libcuda.}

proc cuModuleLoadData*(module: ptr CUmodule; image: pointer): CUresult {.cdecl,
    importc: "cuModuleLoadData", dynlib: libcuda.}

proc cuModuleLoadDataEx*(module: ptr CUmodule; image: pointer; numOptions: cuint;
                        options: ptr CUjitOption; optionValues: ptr pointer): CUresult {.
    cdecl, importc: "cuModuleLoadDataEx", dynlib: libcuda.}

proc cuModuleLoadFatBinary*(module: ptr CUmodule; fatCubin: pointer): CUresult {.cdecl,
    importc: "cuModuleLoadFatBinary", dynlib: libcuda.}

proc cuModuleUnload*(hmod: CUmodule): CUresult {.cdecl, importc: "cuModuleUnload",
    dynlib: libcuda.}

proc cuModuleGetFunction*(hfunc: ptr CUfunction; hmod: CUmodule; name: cstring): CUresult {.
    cdecl, importc: "cuModuleGetFunction", dynlib: libcuda.}

proc cuModuleGetGlobal*(dptr: ptr CUdeviceptr; bytes: ptr csize; hmod: CUmodule;
                       name: cstring): CUresult {.cdecl,
    importc: "cuModuleGetGlobal", dynlib: libcuda.}

proc cuModuleGetTexRef*(pTexRef: ptr CUtexref; hmod: CUmodule; name: cstring): CUresult {.
    cdecl, importc: "cuModuleGetTexRef", dynlib: libcuda.}

proc cuModuleGetSurfRef*(pSurfRef: ptr CUsurfref; hmod: CUmodule; name: cstring): CUresult {.
    cdecl, importc: "cuModuleGetSurfRef", dynlib: libcuda.}
proc cuLinkCreate*(numOptions: cuint; options: ptr CUjitOption;
                  optionValues: ptr pointer; stateOut: ptr CUlinkState): CUresult {.
    cdecl, importc: "cuLinkCreate", dynlib: libcuda.}
proc cuLinkAddData*(state: CUlinkState; `type`: CUjitInputType; data: pointer;
                   size: csize; name: cstring; numOptions: cuint;
                   options: ptr CUjitOption; optionValues: ptr pointer): CUresult {.
    cdecl, importc: "cuLinkAddData", dynlib: libcuda.}
proc cuLinkAddFile*(state: CUlinkState; `type`: CUjitInputType; path: cstring;
                   numOptions: cuint; options: ptr CUjitOption;
                   optionValues: ptr pointer): CUresult {.cdecl,
    importc: "cuLinkAddFile", dynlib: libcuda.}
proc cuLinkComplete*(state: CUlinkState; cubinOut: ptr pointer; sizeOut: ptr csize): CUresult {.
    cdecl, importc: "cuLinkComplete", dynlib: libcuda.}
proc cuLinkDestroy*(state: CUlinkState): CUresult {.cdecl,
    importc: "cuLinkDestroy", dynlib: libcuda.}

proc cuMemGetInfo*(free: ptr csize; total: ptr csize): CUresult {.cdecl,
    importc: "cuMemGetInfo", dynlib: libcuda.}
proc cuMemAlloc*(dptr: ptr CUdeviceptr; bytesize: csize): CUresult {.cdecl,
    importc: "cuMemAlloc", dynlib: libcuda.}
proc cuMemAllocPitch*(dptr: ptr CUdeviceptr; pPitch: ptr csize; widthInBytes: csize;
                     height: csize; elementSizeBytes: cuint): CUresult {.cdecl,
    importc: "cuMemAllocPitch", dynlib: libcuda.}
proc cuMemFree*(dptr: CUdeviceptr): CUresult {.cdecl, importc: "cuMemFree",
    dynlib: libcuda.}
proc cuMemGetAddressRange*(pbase: ptr CUdeviceptr; psize: ptr csize;
                          dptr: CUdeviceptr): CUresult {.cdecl,
    importc: "cuMemGetAddressRange", dynlib: libcuda.}
proc cuMemAllocHost*(pp: ptr pointer; bytesize: csize): CUresult {.cdecl,
    importc: "cuMemAllocHost", dynlib: libcuda.}

proc cuMemFreeHost*(p: pointer): CUresult {.cdecl, importc: "cuMemFreeHost",
                                        dynlib: libcuda.}

proc cuMemHostAlloc*(pp: ptr pointer; bytesize: csize; flags: cuint): CUresult {.cdecl,
    importc: "cuMemHostAlloc", dynlib: libcuda.}

proc cuMemHostGetDevicePointer*(pdptr: ptr CUdeviceptr; p: pointer; flags: cuint): CUresult {.
    cdecl, importc: "cuMemHostGetDevicePointer", dynlib: libcuda.}

proc cuMemHostGetFlags*(pFlags: ptr cuint; p: pointer): CUresult {.cdecl,
    importc: "cuMemHostGetFlags", dynlib: libcuda.}
proc cuMemAllocManaged*(dptr: ptr CUdeviceptr; bytesize: csize; flags: cuint): CUresult {.
    cdecl, importc: "cuMemAllocManaged", dynlib: libcuda.}
proc cuDeviceGetByPCIBusId*(dev: ptr CUdevice; pciBusId: cstring): CUresult {.cdecl,
    importc: "cuDeviceGetByPCIBusId", dynlib: libcuda.}
proc cuDeviceGetPCIBusId*(pciBusId: cstring; len: cint; dev: CUdevice): CUresult {.
    cdecl, importc: "cuDeviceGetPCIBusId", dynlib: libcuda.}
proc cuIpcGetEventHandle*(pHandle: ptr CUipcEventHandle; event: CUevent): CUresult {.
    cdecl, importc: "cuIpcGetEventHandle", dynlib: libcuda.}
proc cuIpcOpenEventHandle*(phEvent: ptr CUevent; handle: CUipcEventHandle): CUresult {.
    cdecl, importc: "cuIpcOpenEventHandle", dynlib: libcuda.}
proc cuIpcGetMemHandle*(pHandle: ptr CUipcMemHandle; dptr: CUdeviceptr): CUresult {.
    cdecl, importc: "cuIpcGetMemHandle", dynlib: libcuda.}
proc cuIpcOpenMemHandle*(pdptr: ptr CUdeviceptr; handle: CUipcMemHandle;
                        flags: cuint): CUresult {.cdecl,
    importc: "cuIpcOpenMemHandle", dynlib: libcuda.}
proc cuIpcCloseMemHandle*(dptr: CUdeviceptr): CUresult {.cdecl,
    importc: "cuIpcCloseMemHandle", dynlib: libcuda.}
proc cuMemHostRegister*(p: pointer; bytesize: csize; flags: cuint): CUresult {.cdecl,
    importc: "cuMemHostRegister", dynlib: libcuda.}
proc cuMemHostUnregister*(p: pointer): CUresult {.cdecl,
    importc: "cuMemHostUnregister", dynlib: libcuda.}
proc cuMemcpy*(dst: CUdeviceptr; src: CUdeviceptr; byteCount: csize): CUresult {.
    cdecl, importc: "cuMemcpy", dynlib: libcuda.}
proc cuMemcpyPeer*(dstDevice: CUdeviceptr; dstContext: CUcontext;
                  srcDevice: CUdeviceptr; srcContext: CUcontext; byteCount: csize): CUresult {.
    cdecl, importc: "cuMemcpyPeer", dynlib: libcuda.}
proc cuMemcpyHtoD*(dstDevice: CUdeviceptr; srcHost: pointer; byteCount: csize): CUresult {.
    cdecl, importc: "cuMemcpyHtoD", dynlib: libcuda.}
proc cuMemcpyDtoH*(dstHost: pointer; srcDevice: CUdeviceptr; byteCount: csize): CUresult {.
    cdecl, importc: "cuMemcpyDtoH", dynlib: libcuda.}
proc cuMemcpyDtoD*(dstDevice: CUdeviceptr; srcDevice: CUdeviceptr; byteCount: csize): CUresult {.
    cdecl, importc: "cuMemcpyDtoD", dynlib: libcuda.}
proc cuMemcpyDtoA*(dstArray: CUarray; dstOffset: csize; srcDevice: CUdeviceptr;
                  byteCount: csize): CUresult {.cdecl, importc: "cuMemcpyDtoA",
    dynlib: libcuda.}
proc cuMemcpyAtoD*(dstDevice: CUdeviceptr; srcArray: CUarray; srcOffset: csize;
                  byteCount: csize): CUresult {.cdecl, importc: "cuMemcpyAtoD",
    dynlib: libcuda.}
proc cuMemcpyHtoA*(dstArray: CUarray; dstOffset: csize; srcHost: pointer;
                  byteCount: csize): CUresult {.cdecl, importc: "cuMemcpyHtoA",
    dynlib: libcuda.}
proc cuMemcpyAtoH*(dstHost: pointer; srcArray: CUarray; srcOffset: csize;
                  byteCount: csize): CUresult {.cdecl, importc: "cuMemcpyAtoH",
    dynlib: libcuda.}
proc cuMemcpyAtoA*(dstArray: CUarray; dstOffset: csize; srcArray: CUarray;
                  srcOffset: csize; byteCount: csize): CUresult {.cdecl,
    importc: "cuMemcpyAtoA", dynlib: libcuda.}
proc cuMemcpy2D*(pCopy: ptr Cuda_Memcpy2d): CUresult {.cdecl, importc: "cuMemcpy2D",
    dynlib: libcuda.}
proc cuMemcpy2DUnaligned*(pCopy: ptr Cuda_Memcpy2d): CUresult {.cdecl,
    importc: "cuMemcpy2DUnaligned", dynlib: libcuda.}
proc cuMemcpy3D*(pCopy: ptr Cuda_Memcpy3d): CUresult {.cdecl, importc: "cuMemcpy3D",
    dynlib: libcuda.}
proc cuMemcpy3DPeer*(pCopy: ptr Cuda_Memcpy3d_Peer): CUresult {.cdecl,
    importc: "cuMemcpy3DPeer", dynlib: libcuda.}
proc cuMemcpyAsync*(dst: CUdeviceptr; src: CUdeviceptr; byteCount: csize;
                   hStream: CUstream): CUresult {.cdecl, importc: "cuMemcpyAsync",
    dynlib: libcuda.}
proc cuMemcpyPeerAsync*(dstDevice: CUdeviceptr; dstContext: CUcontext;
                       srcDevice: CUdeviceptr; srcContext: CUcontext;
                       byteCount: csize; hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpyPeerAsync", dynlib: libcuda.}
proc cuMemcpyHtoDAsync*(dstDevice: CUdeviceptr; srcHost: pointer; byteCount: csize;
                       hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpyHtoDAsync", dynlib: libcuda.}
proc cuMemcpyDtoHAsync*(dstHost: pointer; srcDevice: CUdeviceptr; byteCount: csize;
                       hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpyDtoHAsync", dynlib: libcuda.}
proc cuMemcpyDtoDAsync*(dstDevice: CUdeviceptr; srcDevice: CUdeviceptr;
                       byteCount: csize; hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpyDtoDAsync", dynlib: libcuda.}
proc cuMemcpyHtoAAsync*(dstArray: CUarray; dstOffset: csize; srcHost: pointer;
                       byteCount: csize; hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpyHtoAAsync", dynlib: libcuda.}
proc cuMemcpyAtoHAsync*(dstHost: pointer; srcArray: CUarray; srcOffset: csize;
                       byteCount: csize; hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpyAtoHAsync", dynlib: libcuda.}
proc cuMemcpy2DAsync*(pCopy: ptr Cuda_Memcpy2d; hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpy2DAsync", dynlib: libcuda.}
proc cuMemcpy3DAsync*(pCopy: ptr Cuda_Memcpy3d; hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemcpy3DAsync", dynlib: libcuda.}
proc cuMemcpy3DPeerAsync*(pCopy: ptr Cuda_Memcpy3d_Peer; hStream: CUstream): CUresult {.
    cdecl, importc: "cuMemcpy3DPeerAsync", dynlib: libcuda.}
proc cuMemsetD8*(dstDevice: CUdeviceptr; uc: cuchar; n: csize): CUresult {.cdecl,
    importc: "cuMemsetD8", dynlib: libcuda.}
proc cuMemsetD16*(dstDevice: CUdeviceptr; us: cushort; n: csize): CUresult {.cdecl,
    importc: "cuMemsetD16", dynlib: libcuda.}
proc cuMemsetD32*(dstDevice: CUdeviceptr; ui: cuint; n: csize): CUresult {.cdecl,
    importc: "cuMemsetD32", dynlib: libcuda.}
proc cuMemsetD2D8*(dstDevice: CUdeviceptr; dstPitch: csize; uc: cuchar; width: csize;
                  height: csize): CUresult {.cdecl, importc: "cuMemsetD2D8",
    dynlib: libcuda.}
proc cuMemsetD2D16*(dstDevice: CUdeviceptr; dstPitch: csize; us: cushort;
                   width: csize; height: csize): CUresult {.cdecl,
    importc: "cuMemsetD2D16", dynlib: libcuda.}
proc cuMemsetD2D32*(dstDevice: CUdeviceptr; dstPitch: csize; ui: cuint; width: csize;
                   height: csize): CUresult {.cdecl, importc: "cuMemsetD2D32",
    dynlib: libcuda.}
proc cuMemsetD8Async*(dstDevice: CUdeviceptr; uc: cuchar; n: csize; hStream: CUstream): CUresult {.
    cdecl, importc: "cuMemsetD8Async", dynlib: libcuda.}
proc cuMemsetD16Async*(dstDevice: CUdeviceptr; us: cushort; n: csize;
                      hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemsetD16Async", dynlib: libcuda.}
proc cuMemsetD32Async*(dstDevice: CUdeviceptr; ui: cuint; n: csize; hStream: CUstream): CUresult {.
    cdecl, importc: "cuMemsetD32Async", dynlib: libcuda.}
proc cuMemsetD2D8Async*(dstDevice: CUdeviceptr; dstPitch: csize; uc: cuchar;
                       width: csize; height: csize; hStream: CUstream): CUresult {.
    cdecl, importc: "cuMemsetD2D8Async", dynlib: libcuda.}
proc cuMemsetD2D16Async*(dstDevice: CUdeviceptr; dstPitch: csize; us: cushort;
                        width: csize; height: csize; hStream: CUstream): CUresult {.
    cdecl, importc: "cuMemsetD2D16Async", dynlib: libcuda.}
proc cuMemsetD2D32Async*(dstDevice: CUdeviceptr; dstPitch: csize; ui: cuint;
                        width: csize; height: csize; hStream: CUstream): CUresult {.
    cdecl, importc: "cuMemsetD2D32Async", dynlib: libcuda.}
proc cuArrayCreate*(pHandle: ptr CUarray;
                   pAllocateArray: ptr Cuda_Array_Descriptor): CUresult {.cdecl,
    importc: "cuArrayCreate", dynlib: libcuda.}
proc cuArrayGetDescriptor*(pArrayDescriptor: ptr Cuda_Array_Descriptor;
                          hArray: CUarray): CUresult {.cdecl,
    importc: "cuArrayGetDescriptor", dynlib: libcuda.}

proc cuArrayDestroy*(hArray: CUarray): CUresult {.cdecl, importc: "cuArrayDestroy",
    dynlib: libcuda.}
proc cuArray3DCreate*(pHandle: ptr CUarray;
                     pAllocateArray: ptr Cuda_Array3d_Descriptor): CUresult {.
    cdecl, importc: "cuArray3DCreate", dynlib: libcuda.}
proc cuArray3DGetDescriptor*(pArrayDescriptor: ptr Cuda_Array3d_Descriptor;
                            hArray: CUarray): CUresult {.cdecl,
    importc: "cuArray3DGetDescriptor", dynlib: libcuda.}
proc cuMipmappedArrayCreate*(pHandle: ptr CUmipmappedArray;
                            pMipmappedArrayDesc: ptr Cuda_Array3d_Descriptor;
                            numMipmapLevels: cuint): CUresult {.cdecl,
    importc: "cuMipmappedArrayCreate", dynlib: libcuda.}
proc cuMipmappedArrayGetLevel*(pLevelArray: ptr CUarray;
                              hMipmappedArray: CUmipmappedArray; level: cuint): CUresult {.
    cdecl, importc: "cuMipmappedArrayGetLevel", dynlib: libcuda.}
proc cuMipmappedArrayDestroy*(hMipmappedArray: CUmipmappedArray): CUresult {.
    cdecl, importc: "cuMipmappedArrayDestroy", dynlib: libcuda.}

proc cuPointerGetAttribute*(data: pointer; attribute: CUpointerAttribute;
                           `ptr`: CUdeviceptr): CUresult {.cdecl,
    importc: "cuPointerGetAttribute", dynlib: libcuda.}
proc cuMemPrefetchAsync*(devPtr: CUdeviceptr; count: csize; dstDevice: CUdevice;
                        hStream: CUstream): CUresult {.cdecl,
    importc: "cuMemPrefetchAsync", dynlib: libcuda.}
proc cuMemAdvise*(devPtr: CUdeviceptr; count: csize; advice: CUmemAdvise;
                 device: CUdevice): CUresult {.cdecl, importc: "cuMemAdvise",
    dynlib: libcuda.}
proc cuMemRangeGetAttribute*(data: pointer; dataSize: csize;
                            attribute: CUmemRangeAttribute; devPtr: CUdeviceptr;
                            count: csize): CUresult {.cdecl,
    importc: "cuMemRangeGetAttribute", dynlib: libcuda.}
proc cuMemRangeGetAttributes*(data: ptr pointer; dataSizes: ptr csize;
                             attributes: ptr CUmemRangeAttribute;
                             numAttributes: csize; devPtr: CUdeviceptr;
                             count: csize): CUresult {.cdecl,
    importc: "cuMemRangeGetAttributes", dynlib: libcuda.}
proc cuPointerSetAttribute*(value: pointer; attribute: CUpointerAttribute;
                           `ptr`: CUdeviceptr): CUresult {.cdecl,
    importc: "cuPointerSetAttribute", dynlib: libcuda.}
proc cuPointerGetAttributes*(numAttributes: cuint;
                            attributes: ptr CUpointerAttribute;
                            data: ptr pointer; `ptr`: CUdeviceptr): CUresult {.
    cdecl, importc: "cuPointerGetAttributes", dynlib: libcuda.}

proc cuStreamCreate*(phStream: ptr CUstream; flags: cuint): CUresult {.cdecl,
    importc: "cuStreamCreate", dynlib: libcuda.}

proc cuStreamCreateWithPriority*(phStream: ptr CUstream; flags: cuint; priority: cint): CUresult {.
    cdecl, importc: "cuStreamCreateWithPriority", dynlib: libcuda.}

proc cuStreamGetPriority*(hStream: CUstream; priority: ptr cint): CUresult {.cdecl,
    importc: "cuStreamGetPriority", dynlib: libcuda.}

proc cuStreamGetFlags*(hStream: CUstream; flags: ptr cuint): CUresult {.cdecl,
    importc: "cuStreamGetFlags", dynlib: libcuda.}

proc cuStreamWaitEvent*(hStream: CUstream; hEvent: CUevent; flags: cuint): CUresult {.
    cdecl, importc: "cuStreamWaitEvent", dynlib: libcuda.}

proc cuStreamAddCallback*(hStream: CUstream; callback: CUstreamCallback;
                         userData: pointer; flags: cuint): CUresult {.cdecl,
    importc: "cuStreamAddCallback", dynlib: libcuda.}
proc cuStreamAttachMemAsync*(hStream: CUstream; dptr: CUdeviceptr; length: csize;
                            flags: cuint): CUresult {.cdecl,
    importc: "cuStreamAttachMemAsync", dynlib: libcuda.}

proc cuStreamQuery*(hStream: CUstream): CUresult {.cdecl, importc: "cuStreamQuery",
    dynlib: libcuda.}

proc cuStreamSynchronize*(hStream: CUstream): CUresult {.cdecl,
    importc: "cuStreamSynchronize", dynlib: libcuda.}
proc cuStreamDestroy*(hStream: CUstream): CUresult {.cdecl,
    importc: "cuStreamDestroy", dynlib: libcuda.}

proc cuEventCreate*(phEvent: ptr CUevent; flags: cuint): CUresult {.cdecl,
    importc: "cuEventCreate", dynlib: libcuda.}

proc cuEventRecord*(hEvent: CUevent; hStream: CUstream): CUresult {.cdecl,
    importc: "cuEventRecord", dynlib: libcuda.}

proc cuEventQuery*(hEvent: CUevent): CUresult {.cdecl, importc: "cuEventQuery",
    dynlib: libcuda.}

proc cuEventSynchronize*(hEvent: CUevent): CUresult {.cdecl,
    importc: "cuEventSynchronize", dynlib: libcuda.}
proc cuEventDestroy*(hEvent: CUevent): CUresult {.cdecl, importc: "cuEventDestroy",
    dynlib: libcuda.}

proc cuEventElapsedTime*(pMilliseconds: ptr cfloat; hStart: CUevent; hEnd: CUevent): CUresult {.
    cdecl, importc: "cuEventElapsedTime", dynlib: libcuda.}
proc cuStreamWaitValue32*(stream: CUstream; `addr`: CUdeviceptr; value: uint32;
                         flags: cuint): CUresult {.cdecl,
    importc: "cuStreamWaitValue32", dynlib: libcuda.}
proc cuStreamWriteValue32*(stream: CUstream; `addr`: CUdeviceptr; value: uint32;
                          flags: cuint): CUresult {.cdecl,
    importc: "cuStreamWriteValue32", dynlib: libcuda.}
proc cuStreamBatchMemOp*(stream: CUstream; count: cuint;
                        paramArray: ptr CUstreamBatchMemOpParams; flags: cuint): CUresult {.
    cdecl, importc: "cuStreamBatchMemOp", dynlib: libcuda.}

proc cuFuncGetAttribute*(pi: ptr cint; attrib: CUfunctionAttribute; hfunc: CUfunction): CUresult {.
    cdecl, importc: "cuFuncGetAttribute", dynlib: libcuda.}

proc cuFuncSetCacheConfig*(hfunc: CUfunction; config: CUfuncCache): CUresult {.cdecl,
    importc: "cuFuncSetCacheConfig", dynlib: libcuda.}
proc cuFuncSetSharedMemConfig*(hfunc: CUfunction; config: CUsharedconfig): CUresult {.
    cdecl, importc: "cuFuncSetSharedMemConfig", dynlib: libcuda.}
proc cuLaunchKernel*(f: CUfunction; gridDimX: cuint; gridDimY: cuint;
                    gridDimZ: cuint; blockDimX: cuint; blockDimY: cuint;
                    blockDimZ: cuint; sharedMemBytes: cuint; hStream: CUstream;
                    kernelParams: ptr pointer; extra: ptr pointer): CUresult {.cdecl,
    importc: "cuLaunchKernel", dynlib: libcuda.}

proc cuFuncSetBlockShape*(hfunc: CUfunction; x: cint; y: cint; z: cint): CUresult {.cdecl,
    importc: "cuFuncSetBlockShape", dynlib: libcuda.}

proc cuFuncSetSharedSize*(hfunc: CUfunction; bytes: cuint): CUresult {.cdecl,
    importc: "cuFuncSetSharedSize", dynlib: libcuda.}

proc cuParamSetSize*(hfunc: CUfunction; numbytes: cuint): CUresult {.cdecl,
    importc: "cuParamSetSize", dynlib: libcuda.}

proc cuParamSeti*(hfunc: CUfunction; offset: cint; value: cuint): CUresult {.cdecl,
    importc: "cuParamSeti", dynlib: libcuda.}

proc cuParamSetf*(hfunc: CUfunction; offset: cint; value: cfloat): CUresult {.cdecl,
    importc: "cuParamSetf", dynlib: libcuda.}

proc cuParamSetv*(hfunc: CUfunction; offset: cint; `ptr`: pointer; numbytes: cuint): CUresult {.
    cdecl, importc: "cuParamSetv", dynlib: libcuda.}

proc cuLaunch*(f: CUfunction): CUresult {.cdecl, importc: "cuLaunch", dynlib: libcuda.}

proc cuLaunchGrid*(f: CUfunction; gridWidth: cint; gridHeight: cint): CUresult {.cdecl,
    importc: "cuLaunchGrid", dynlib: libcuda.}

proc cuLaunchGridAsync*(f: CUfunction; gridWidth: cint; gridHeight: cint;
                       hStream: CUstream): CUresult {.cdecl,
    importc: "cuLaunchGridAsync", dynlib: libcuda.}

proc cuParamSetTexRef*(hfunc: CUfunction; texunit: cint; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuParamSetTexRef", dynlib: libcuda.}

proc cuOccupancyMaxActiveBlocksPerMultiprocessor*(numBlocks: ptr cint;
    `func`: CUfunction; blockSize: cint; dynamicSMemSize: csize): CUresult {.cdecl,
    importc: "cuOccupancyMaxActiveBlocksPerMultiprocessor", dynlib: libcuda.}
proc cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags*(numBlocks: ptr cint;
    `func`: CUfunction; blockSize: cint; dynamicSMemSize: csize; flags: cuint): CUresult {.
    cdecl, importc: "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    dynlib: libcuda.}
proc cuOccupancyMaxPotentialBlockSize*(minGridSize: ptr cint; blockSize: ptr cint;
                                      `func`: CUfunction;
    blockSizeToDynamicSMemSize: CUoccupancyB2DSize; dynamicSMemSize: csize;
                                      blockSizeLimit: cint): CUresult {.cdecl,
    importc: "cuOccupancyMaxPotentialBlockSize", dynlib: libcuda.}
proc cuOccupancyMaxPotentialBlockSizeWithFlags*(minGridSize: ptr cint;
    blockSize: ptr cint; `func`: CUfunction;
    blockSizeToDynamicSMemSize: CUoccupancyB2DSize; dynamicSMemSize: csize;
    blockSizeLimit: cint; flags: cuint): CUresult {.cdecl,
    importc: "cuOccupancyMaxPotentialBlockSizeWithFlags", dynlib: libcuda.}

proc cuTexRefSetArray*(hTexRef: CUtexref; hArray: CUarray; flags: cuint): CUresult {.
    cdecl, importc: "cuTexRefSetArray", dynlib: libcuda.}

proc cuTexRefSetMipmappedArray*(hTexRef: CUtexref;
                               hMipmappedArray: CUmipmappedArray; flags: cuint): CUresult {.
    cdecl, importc: "cuTexRefSetMipmappedArray", dynlib: libcuda.}
proc cuTexRefSetAddress*(byteOffset: ptr csize; hTexRef: CUtexref;
                        dptr: CUdeviceptr; bytes: csize): CUresult {.cdecl,
    importc: "cuTexRefSetAddress", dynlib: libcuda.}
proc cuTexRefSetAddress2D*(hTexRef: CUtexref; desc: ptr Cuda_Array_Descriptor;
                          dptr: CUdeviceptr; pitch: csize): CUresult {.cdecl,
    importc: "cuTexRefSetAddress2D", dynlib: libcuda.}

proc cuTexRefSetFormat*(hTexRef: CUtexref; fmt: CUarrayFormat;
                       numPackedComponents: cint): CUresult {.cdecl,
    importc: "cuTexRefSetFormat", dynlib: libcuda.}

proc cuTexRefSetAddressMode*(hTexRef: CUtexref; dim: cint; am: CUaddressMode): CUresult {.
    cdecl, importc: "cuTexRefSetAddressMode", dynlib: libcuda.}

proc cuTexRefSetFilterMode*(hTexRef: CUtexref; fm: CUfilterMode): CUresult {.cdecl,
    importc: "cuTexRefSetFilterMode", dynlib: libcuda.}

proc cuTexRefSetMipmapFilterMode*(hTexRef: CUtexref; fm: CUfilterMode): CUresult {.
    cdecl, importc: "cuTexRefSetMipmapFilterMode", dynlib: libcuda.}

proc cuTexRefSetMipmapLevelBias*(hTexRef: CUtexref; bias: cfloat): CUresult {.cdecl,
    importc: "cuTexRefSetMipmapLevelBias", dynlib: libcuda.}

proc cuTexRefSetMipmapLevelClamp*(hTexRef: CUtexref; minMipmapLevelClamp: cfloat;
                                 maxMipmapLevelClamp: cfloat): CUresult {.cdecl,
    importc: "cuTexRefSetMipmapLevelClamp", dynlib: libcuda.}

proc cuTexRefSetMaxAnisotropy*(hTexRef: CUtexref; maxAniso: cuint): CUresult {.cdecl,
    importc: "cuTexRefSetMaxAnisotropy", dynlib: libcuda.}

proc cuTexRefSetBorderColor*(hTexRef: CUtexref; pBorderColor: ptr cfloat): CUresult {.
    cdecl, importc: "cuTexRefSetBorderColor", dynlib: libcuda.}

proc cuTexRefSetFlags*(hTexRef: CUtexref; flags: cuint): CUresult {.cdecl,
    importc: "cuTexRefSetFlags", dynlib: libcuda.}
proc cuTexRefGetAddress*(pdptr: ptr CUdeviceptr; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuTexRefGetAddress", dynlib: libcuda.}

proc cuTexRefGetArray*(phArray: ptr CUarray; hTexRef: CUtexref): CUresult {.cdecl,
    importc: "cuTexRefGetArray", dynlib: libcuda.}

proc cuTexRefGetMipmappedArray*(phMipmappedArray: ptr CUmipmappedArray;
                               hTexRef: CUtexref): CUresult {.cdecl,
    importc: "cuTexRefGetMipmappedArray", dynlib: libcuda.}

proc cuTexRefGetAddressMode*(pam: ptr CUaddressMode; hTexRef: CUtexref; dim: cint): CUresult {.
    cdecl, importc: "cuTexRefGetAddressMode", dynlib: libcuda.}

proc cuTexRefGetFilterMode*(pfm: ptr CUfilterMode; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuTexRefGetFilterMode", dynlib: libcuda.}

proc cuTexRefGetFormat*(pFormat: ptr CUarrayFormat; pNumChannels: ptr cint;
                       hTexRef: CUtexref): CUresult {.cdecl,
    importc: "cuTexRefGetFormat", dynlib: libcuda.}

proc cuTexRefGetMipmapFilterMode*(pfm: ptr CUfilterMode; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuTexRefGetMipmapFilterMode", dynlib: libcuda.}

proc cuTexRefGetMipmapLevelBias*(pbias: ptr cfloat; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuTexRefGetMipmapLevelBias", dynlib: libcuda.}

proc cuTexRefGetMipmapLevelClamp*(pminMipmapLevelClamp: ptr cfloat;
                                 pmaxMipmapLevelClamp: ptr cfloat;
                                 hTexRef: CUtexref): CUresult {.cdecl,
    importc: "cuTexRefGetMipmapLevelClamp", dynlib: libcuda.}

proc cuTexRefGetMaxAnisotropy*(pmaxAniso: ptr cint; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuTexRefGetMaxAnisotropy", dynlib: libcuda.}

proc cuTexRefGetBorderColor*(pBorderColor: ptr cfloat; hTexRef: CUtexref): CUresult {.
    cdecl, importc: "cuTexRefGetBorderColor", dynlib: libcuda.}

proc cuTexRefGetFlags*(pFlags: ptr cuint; hTexRef: CUtexref): CUresult {.cdecl,
    importc: "cuTexRefGetFlags", dynlib: libcuda.}

proc cuTexRefCreate*(pTexRef: ptr CUtexref): CUresult {.cdecl,
    importc: "cuTexRefCreate", dynlib: libcuda.}

proc cuTexRefDestroy*(hTexRef: CUtexref): CUresult {.cdecl,
    importc: "cuTexRefDestroy", dynlib: libcuda.}

proc cuSurfRefSetArray*(hSurfRef: CUsurfref; hArray: CUarray; flags: cuint): CUresult {.
    cdecl, importc: "cuSurfRefSetArray", dynlib: libcuda.}

proc cuSurfRefGetArray*(phArray: ptr CUarray; hSurfRef: CUsurfref): CUresult {.cdecl,
    importc: "cuSurfRefGetArray", dynlib: libcuda.}

proc cuTexObjectCreate*(pTexObject: ptr CUtexObject;
                       pResDesc: ptr Cuda_Resource_Desc;
                       pTexDesc: ptr Cuda_Texture_Desc;
                       pResViewDesc: ptr Cuda_Resource_View_Desc): CUresult {.
    cdecl, importc: "cuTexObjectCreate", dynlib: libcuda.}
proc cuTexObjectDestroy*(texObject: CUtexObject): CUresult {.cdecl,
    importc: "cuTexObjectDestroy", dynlib: libcuda.}
proc cuTexObjectGetResourceDesc*(pResDesc: ptr Cuda_Resource_Desc;
                                texObject: CUtexObject): CUresult {.cdecl,
    importc: "cuTexObjectGetResourceDesc", dynlib: libcuda.}
proc cuTexObjectGetTextureDesc*(pTexDesc: ptr Cuda_Texture_Desc;
                               texObject: CUtexObject): CUresult {.cdecl,
    importc: "cuTexObjectGetTextureDesc", dynlib: libcuda.}
proc cuTexObjectGetResourceViewDesc*(pResViewDesc: ptr Cuda_Resource_View_Desc;
                                    texObject: CUtexObject): CUresult {.cdecl,
    importc: "cuTexObjectGetResourceViewDesc", dynlib: libcuda.}
proc cuSurfObjectCreate*(pSurfObject: ptr CUsurfObject;
                        pResDesc: ptr Cuda_Resource_Desc): CUresult {.cdecl,
    importc: "cuSurfObjectCreate", dynlib: libcuda.}
proc cuSurfObjectDestroy*(surfObject: CUsurfObject): CUresult {.cdecl,
    importc: "cuSurfObjectDestroy", dynlib: libcuda.}
proc cuSurfObjectGetResourceDesc*(pResDesc: ptr Cuda_Resource_Desc;
                                 surfObject: CUsurfObject): CUresult {.cdecl,
    importc: "cuSurfObjectGetResourceDesc", dynlib: libcuda.}
proc cuDeviceCanAccessPeer*(canAccessPeer: ptr cint; dev: CUdevice;
                           peerDev: CUdevice): CUresult {.cdecl,
    importc: "cuDeviceCanAccessPeer", dynlib: libcuda.}
proc cuDeviceGetP2PAttribute*(value: ptr cint; attrib: CUdeviceP2PAttribute;
                             srcDevice: CUdevice; dstDevice: CUdevice): CUresult {.
    cdecl, importc: "cuDeviceGetP2PAttribute", dynlib: libcuda.}
proc cuCtxEnablePeerAccess*(peerContext: CUcontext; flags: cuint): CUresult {.cdecl,
    importc: "cuCtxEnablePeerAccess", dynlib: libcuda.}
proc cuCtxDisablePeerAccess*(peerContext: CUcontext): CUresult {.cdecl,
    importc: "cuCtxDisablePeerAccess", dynlib: libcuda.}

proc cuGraphicsUnregisterResource*(resource: CUgraphicsResource): CUresult {.cdecl,
    importc: "cuGraphicsUnregisterResource", dynlib: libcuda.}

proc cuGraphicsSubResourceGetMappedArray*(pArray: ptr CUarray;
    resource: CUgraphicsResource; arrayIndex: cuint; mipLevel: cuint): CUresult {.
    cdecl, importc: "cuGraphicsSubResourceGetMappedArray", dynlib: libcuda.}
proc cuGraphicsResourceGetMappedMipmappedArray*(
    pMipmappedArray: ptr CUmipmappedArray; resource: CUgraphicsResource): CUresult {.
    cdecl, importc: "cuGraphicsResourceGetMappedMipmappedArray", dynlib: libcuda.}
proc cuGraphicsResourceGetMappedPointer*(pDevPtr: ptr CUdeviceptr;
    pSize: ptr csize; resource: CUgraphicsResource): CUresult {.cdecl,
    importc: "cuGraphicsResourceGetMappedPointer", dynlib: libcuda.}

proc cuGraphicsResourceSetMapFlags*(resource: CUgraphicsResource; flags: cuint): CUresult {.
    cdecl, importc: "cuGraphicsResourceSetMapFlags", dynlib: libcuda.}

proc cuGraphicsMapResources*(count: cuint; resources: ptr CUgraphicsResource;
                            hStream: CUstream): CUresult {.cdecl,
    importc: "cuGraphicsMapResources", dynlib: libcuda.}

proc cuGraphicsUnmapResources*(count: cuint; resources: ptr CUgraphicsResource;
                              hStream: CUstream): CUresult {.cdecl,
    importc: "cuGraphicsUnmapResources", dynlib: libcuda.}

proc cuGetExportTable*(ppExportTable: ptr pointer; pExportTableId: ptr CUuuid): CUresult {.
    cdecl, importc: "cuGetExportTable", dynlib: libcuda.}

# proc cuMemHostRegister*(p: pointer; bytesize: csize; flags: cuint): CUresult {.cdecl,
#     importc: "cuMemHostRegister", dynlib: libcuda.}
# proc cuGraphicsResourceSetMapFlags*(resource: CUgraphicsResource; flags: cuint): CUresult {.
#     cdecl, importc: "cuGraphicsResourceSetMapFlags", dynlib: libcuda.}
# proc cuLinkCreate*(numOptions: cuint; options: ptr CUjitOption;
#                   optionValues: ptr pointer; stateOut: ptr CUlinkState): CUresult {.
#     cdecl, importc: "cuLinkCreate", dynlib: libcuda.}
# proc cuLinkAddData*(state: CUlinkState; `type`: CUjitInputType; data: pointer;
#                    size: csize; name: cstring; numOptions: cuint;
#                    options: ptr CUjitOption; optionValues: ptr pointer): CUresult {.
#     cdecl, importc: "cuLinkAddData", dynlib: libcuda.}
# proc cuLinkAddFile*(state: CUlinkState; `type`: CUjitInputType; path: cstring;
#                    numOptions: cuint; options: ptr CUjitOption;
#                    optionValues: ptr pointer): CUresult {.cdecl,
#     importc: "cuLinkAddFile", dynlib: libcuda.}
proc cuTexRefSetAddress2D_v2*(hTexRef: CUtexref; desc: ptr Cuda_Array_Descriptor;
                         dptr: CUdeviceptr; pitch: csize): CUresult {.cdecl,
    importc: "cuTexRefSetAddress2D_v2", dynlib: libcuda.}
