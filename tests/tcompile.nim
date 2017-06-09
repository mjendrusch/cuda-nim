import cuda, unittest, sequtils

const kernelCode = """
// KERNEL SECTION:
extern "C" __global__ void axpyKernel(double* a, double* x, double* y) {
  int idx_255 = threadIdx.x;
  x[idx_255] = ((a[idx_255] * x[idx_255]) + y[idx_255]);
}
"""

suite "compilation":
  test "compile and compare ptx":
    proc test =
      let prog = newProgram(kernelCode)
      prog.compile
      let ptx = prog.ptx

    test()

  test "run compiled ptx":
    proc test =
      # Compile the program
      let prog = newProgram(kernelCode)
      prog.compile
      let ptx = prog.ptx
      # Set up and run the kernel
      cuda.init()
      var
        devCount = getDeviceCount()
        dev = getDevice(0)
        ctx = newContext(dev)
        module = newModuleFromData(ptx)
        kernel = module.getKernel("axpyKernel")
        sq1 = newSeqWith(1024, 1.0)
        sq2 = newSeqWith(1024, 2.0)
        sq3 = newSeqWith(1024, 3.0)
        x, y, a = deviceAlloc(uint sizeOf(float) * 1024)
        args = newSeq[pointer]()
      args.add(a.addr)
      args.add(x.addr)
      args.add(y.addr)
      copyMem(x, sq1[0].addr, 1024 * sizeOf(float))
      copyMem(y, sq2[0].addr, 1024 * sizeOf(float))
      copyMem(a, sq3[0].addr, 1024 * sizeOf(float))

      kernel.launch(Dim(x: 1, y: 1, z: 1), Dim(x: 1024, y: 1, z: 1), args, 0)
      copyMem(sq1[0].addr, x, 1024 * sizeOf(float))
      copyMem(sq2[0].addr, y, 1024 * sizeOf(float))
      copyMem(sq3[0].addr, a, 1024 * sizeOf(float))
      for idx in 0 ..< 1024:
        check sq1[idx] == 5.0
        check sq2[idx] == 2.0
        check sq3[idx] == 3.0

    test()
