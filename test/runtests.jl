# compute models in parallel: multiprocess
addprocs(Sys.CPU_CORES) # add a worker process per core
print_with_color(:white, "Setup:\n")
println("  > Using $(nprocs()-1) worker processes")

using PBupsModel
using Base.Test

# write your own tests here
@test 1 == 2
