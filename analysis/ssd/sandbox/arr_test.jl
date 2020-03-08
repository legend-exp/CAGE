
"""
try some operations on a 2d array, with a kernel function that randomizes 
each element.  the goal is to find an order of operations that will allow the
kernel function to be compiled and then run at "full speed", i.e. what we get 
when we run a command twice in the REPL.
"""
function main()
    nrow, ncol, niter = 10, 20, 100000
    arr = init_array(nrow, ncol)
    
    randomize_once!(nrow, ncol, niter)
    
    for j = 1:niter
        randomize_once!(arr)
        if verbose
            print("NITER: ",j, "\n")
            display(arr)
        end
    end
    
    # randomize_brute!(nrow, ncol, niter)
    
    # this is like "run once w/ dummy arguments, then a bunch of times"
    # randomize_once!(arr)
    # randomize_lots!(arr, niter)
    
    # hmm.  both the above give the same speed, and both have to be run twice from the REPL to get the speed boost
    
    # try to specify types of everything to make it run faster
    # randomize_withtypes!(nrow, ncol, niter)
end


function init_array(nrow, ncol)
    # play w/ a few different ways to initialize the 2d array

    # method 1 - no type declared
    # arr = reshape([],0,2)
    
    # method 2 - with type
    # arr = Array{Float64}(undef, 2, 2)
    
    # method 3 - from random numbers
    arr = rand(0:.1:1, nrow, ncol) # value, nrow, ncol

    # remember, Julia is ONE-INDEXED
    # arr[1,1] = 1

    # this prints everything - probably a bad idea
    # print(arr)
    
    # this is what the REPL is doing (from stackoverflow)
    # display(arr)
    
    return arr
end

"""
kernel function - representing an arbitrary array computation,
and explicitly not vectorized or optimized.
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
randomize the array a bunch of times
"""
function randomize_lots!(arr, niter, verbose=false)
    
end


"""
declare and randomize the array in the same function
"""
function randomize_brute!(nrow, ncol, niter, verbose=false)
    arr = rand(0:.1:1, nrow, ncol)
    
    for n in 1:niter
        for i in 1:nrow
            for j in 1:ncol
                arr[i, j] = rand(0:.1:1)
            end
        end
        if verbose
            print("NITER: ",n, "\n")
            display(arr)
        end
    end
end


"""
try to get a speed boost by specifying the type of everything
"""
function randomize_withtypes!(nrow::Int, ncol::Int, niter::Int)::Array{Float64, 2}
    arr = Array{Float64, 2}(undef, nrow, ncol)
    
    for n in 1:niter
        for i in 1:nrow
            for j in 1:ncol
                arr[i, j] = rand(0:.1:1)
            end
        end
    end
    print("done")
    return arr
end


main()