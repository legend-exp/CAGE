#=
try an operation on a 2d array with a kernel function.

the goal is to find an order of function calls that will allow the
processing loop and kernel function to be compiled and run at "full speed",
i.e. what we get when we run a command twice in the REPL.

i think this illustrates how julia's JIT compiling works?
=#

"""
kernel function - representing an arbitrary array computation,
and not vectorized or optimized on purpose, to leave room for improvement ;-)
"""
function randomize_once!(arr)
    nrow, ncol = size(arr)
    for i in 1:nrow
        for j in 1:ncol
            arr[i, j] = rand(0:.1:1)
        end
    end
end

"""
call the kernel function repeatedly - representing a processing loop
"""
function randomize_arr(nrow, ncol, niter)
    arr = rand(0:.1:1, nrow, ncol) 
    for j = 1:niter
        randomize_once!(arr)
    end
end

# print(ARGS)

# running w/ this line first should make the main loop go faster, right?
randomize_arr(1, 1, 1) 

# this is the main loop
@time randomize_arr(100, 200, 1000)
