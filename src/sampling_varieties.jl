using HomotopyContinuation, DynamicPolynomials, LinearAlgebra, IterTools

function compute_bottlenecks(F)
    # Returns an array of all real non-singular 2-bottlenecks of the variety F
    x = variables(F)
    n = length(x)
    d=length(F) # codimension of variety
    k = n-d # dimension of variety
    @polyvar y[1:n] lambda[1:d] mu[1:d]; # lagrange multipliers

    # Compute the bottlenecks
    grad = differentiate(F, x)
    G = subs(F, x => y)
    grady = subs(grad, x => y)

    system = [F; G; map(j -> x[j]-y[j]-dot(lambda, grad[:, j]), 1:n); map(j -> x[j]-y[j]-dot(mu, grady[:, j]), 1:n)]
    result = solve(system, start_system = :polyhedral)
    bottlenecks = map(s -> (s[1:n], s[n+1:2*n]), real_solutions(nonsingular(result)))
    return bottlenecks
end

function get_bottleneck_lengths(bottlenecks)
    # Returns a tuple [radius, bottleneck] of the Euclidean radius of a list of 2-bottlenecks
    bn_lengths = sort!(map(b -> (norm(b[1]-b[2])/2, b), bottlenecks), by = a -> a[1])
    return bn_lengths
end

function sampling_bottlenecks(F)
    # F - system, n - ambient dimension, x - array of variables of F, epsilon - density
    x = variables(F)
    n = length(x)
    @polyvar y[1:n] p[1:n] gamma[1:n]
    d=length(F) # codimension of variety
    k = n-d # dimension of variety
    @polyvar lambda[1:d] # lagrange multipliers

    bottlenecks = compute_bottlenecks(F)
    δ = get_bottleneck_lengths(bottlenecks)[1][1] # grid size

    gradx = differentiate(F, x)

    # Compute the bounding box by computing the EDD starting from the center of the largest bottleneck

    q = [1 for _ in 1:n]
    system = [F; map(j -> x[j]-q[j]-dot(lambda, gradx[:, j]), 1:n)]
    result = solve(system, start_system = :polyhedral)


    # Extract farthest point from q to X and use as box length

    critical_points = sort!(map(c -> (norm(c[1:n]-q), c[1:n]), real_solutions(nonsingular(result))), by = a -> a[1])
    b = critical_points[end][1]
    indices = [i for i in -b:δ:b];


    # Compute basic sample

    samples = []
    counter = 0

    start_time = time_ns()
    for s in IterTools.subsets(1:n, k)
        Ft = System([F; map(i -> x[s[i]]-p[i]-q[s[i]], 1:k)]; parameters=p[1:k])
        p₀ = randn(ComplexF64, k)
        result_p₀ = solve(Ft, target_parameters = p₀)
        S_p₀ = solutions(result_p₀)

        slice_samples = solve(
            Ft,
            S_p₀;
            start_parameters =  p₀,
            target_parameters = [map(j -> indices[p1[j]], 1:k) for p1 in Iterators.product(map(j-> 1:length(indices), s)...)],
            transform_result = (r,p) -> real_solutions(r),
            flatten = true
        )
        samples = vcat(samples, slice_samples)
    end


    # Compute extra sample

    extra_samples = []
    extra_counter = 0

    start_time = time_ns()
    for l in 1:k-1
        for s in IterTools.subsets(1:n, l)
            Ft = [F; map(i -> x[s[i]]-p[i]-q[s[i]], 1:l)]
            grad = differentiate(Ft, x)
            system = System([Ft; map(j -> x[j]-y[j]-dot(gamma[1:n-k+l], grad[:, j]), 1:n)]; parameters=[y; p[1:l]])
            p₀ = randn(ComplexF64, n+l)
            result_p₀ = solve(system, target_parameters = p₀)
            S_p₀ = solutions(result_p₀)

            slice_samples = solve(
                system,
                S_p₀;
                start_parameters =  p₀,
                target_parameters = [[randn(Float64, n); map(j -> indices[p1[j]], 1:l)] for p1 in Iterators.product(map(j-> 1:length(indices), s)...)],
                transform_result = (r,p) -> real_solutions(r),
                flatten = true
            )
            extra_samples = vcat(extra_samples, slice_samples)
        end
    end

    return vcat(samples, extra_samples)
end

function sampling_fixed_density(F,epsilon)
    # F - system, n - ambient dimension, x - array of variables of F, epsilon - density
    x = variables(F)
    n = length(x) 
    @polyvar y[1:n] p[1:n] gamma[1:n]
    d=length(F) # codimension of variety
    k = n-d # dimension of variety
    @polyvar lambda[1:d] # lagrange multipliers
    
    gradx = differentiate(F, x)
    
    δ = epsilon/(2*sqrt(n)) # grid size
    
    
    # Compute the bounding box by computing the EDD starting from the center of the largest bottleneck

    q = [1 for _ in 1:n]
    system = [F; map(j -> x[j]-q[j]-dot(lambda, gradx[:, j]), 1:n)]
    result = solve(system, start_system = :polyhedral)
    
    
    # Extract farthest point from q to X and use as box length

    critical_points = sort!(map(c -> (norm(c[1:n]-q), c[1:n]), real_solutions(nonsingular(result))), by = a -> a[1])
    b = critical_points[end][1]
    indices = [i for i in -b:δ:b];
    
    
    # Compute basic sample

    samples = []
    counter = 0

    start_time = time_ns()
    for s in IterTools.subsets(1:n, k)
        Ft = System([F; map(i -> x[s[i]]-p[i]-q[s[i]], 1:k)]; parameters=p[1:k])
        p₀ = randn(ComplexF64, k)
        result_p₀ = solve(Ft, target_parameters = p₀)
        S_p₀ = solutions(result_p₀)

        slice_samples = solve(
        Ft,
        S_p₀;
        start_parameters =  p₀,
        target_parameters = [map(j -> indices[p1[j]], 1:k) for p1 in Iterators.product(map(j-> 1:length(indices), s)...)],
        transform_result = (r,p) -> real_solutions(r),
        flatten = true
        )
        samples = vcat(samples, slice_samples)
    end
    
    
    # Compute extra sample

    extra_samples = []
    extra_counter = 0

    start_time = time_ns()
    for l in 1:k-1
        for s in IterTools.subsets(1:n, l)
            Ft = [F; map(i -> x[s[i]]-p[i]-q[s[i]], 1:l)]
            grad = differentiate(Ft, x)
            system = System([Ft; map(j -> x[j]-y[j]-dot(gamma[1:n-k+l], grad[:, j]), 1:n)]; parameters=[y; p[1:l]])
            p₀ = randn(ComplexF64, n+l)
            result_p₀ = solve(system, target_parameters = p₀)
            S_p₀ = solutions(result_p₀)

            slice_samples = solve(
            system,
            S_p₀;
            start_parameters =  p₀,
            target_parameters = [[randn(Float64, n); map(j -> indices[p1[j]], 1:l)] for p1 in Iterators.product(map(j-> 1:length(indices), s)...)],
            transform_result = (r,p) -> real_solutions(r),
            flatten = true
            )
            extra_samples = vcat(extra_samples, slice_samples)
        end
    end
    
    return vcat(samples, extra_samples)
end
