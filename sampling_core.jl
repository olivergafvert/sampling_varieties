using HomotopyContinuation, DynamicPolynomials, LinearAlgebra, IterTools, Ripserer, Plots

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
        Ft = [F; map(i -> x[s[i]]-p[i]-q[s[i]], 1:k)]
        p₀ = randn(ComplexF64, k)
        F_p₀ = subs(Ft, p[1:k] => p₀)
        result_p₀ = solve(F_p₀)
        S_p₀ = solutions(result_p₀)

        # Construct the PathTracker
        tracker = HomotopyContinuation.pathtracker(Ft; parameters=p[1:k], generic_parameters=p₀)
        for p1 in Iterators.product(map(j-> 1:length(indices), s)...)
            counter += length(S_p₀)
            for s1 in S_p₀
                result = track(tracker, s1; target_parameters=map(j -> indices[p1[j]], 1:k))
                # check that the tracking was successfull
                if is_success(result) && is_real(result)
                    push!(samples, real(solution(result)))
                end
            end
        end
    end
    
    
    # Compute extra sample

    extra_samples = []
    extra_counter = 0

    start_time = time_ns()
    for l in 1:k-1
        for s in IterTools.subsets(1:n, l)
            Ft = [F; map(i -> x[s[i]]-p[i]-q[s[i]], 1:l)] 
            gradx = differentiate(Ft, x)
            system = [Ft; map(j -> x[j]-y[j]-dot(gamma[1:n-k+l], gradx[:, j]), 1:n)]

            p₀ = randn(ComplexF64, n+l)
            F_p₀ = subs(system, [y; p[1:l]] => p₀)
            result_p₀ = solve(F_p₀)
            S_p₀ = solutions(result_p₀)

            # Construct the PathTracker
            tracker = HomotopyContinuation.pathtracker(system; parameters=[y; p[1:l]], generic_parameters=p₀)
            for p1 in Iterators.product(map(j-> 1:length(indices), s)...)
                extra_counter += length(S_p₀)
                for s1 in S_p₀
                    result = track(tracker, s1; target_parameters=[randn(Float64, n); map(j -> indices[p1[j]], 1:l)])
                    # check that the tracking was successfull
                    if is_success(result) && is_real(result)
                        push!(extra_samples, real(solution(result))[1:n])
                    end
                end
            end
        end
    end
    
    return vcat(samples, extra_samples)
end

function construct_grid_points(initial,final,number)
    step_size = (final-initial)/(number-1)
    return [initial + i*step_size for i in 0:(number-1)]
end
   
function subsample_with_function(sample,func,norm_func=norm)
    sample = sort(sample,by=func)
    output = []
    while(length(sample) > 0)
        the_current_point = pop!(sample)
        push!(output,the_current_point)
        should_remove = [false for _ in 1:length(sample)]
        current_threshold = func(the_current_point)
        for i in 1:length(sample)
            if norm_func(sample[i]-the_current_point) < current_threshold
                should_remove[i] = 1
            end
        end
        sample = [sample[i] for i in 1:length(sample) if should_remove[i] == false]
    end
    return output
end

function constant_func_curry(the_constant=0) 
    return function(x) return the_constant end
end

function naive_sparsification_tester(F,n,x,target_density,number_of_discretization_points=10)
    range_of_proportions = construct_grid_points(1/number_of_discretization_points,1,number_of_discretization_points)
    original_number_of_points = []
    sparsified_number_of_points = []
    for proportion in range_of_proportions
        sampling_density = target_density*proportion
        the_sample = sampling_fixed_density(F, n, x, sampling_density)
        push!(original_number_of_points,length(the_sample))
        subsampling_allowance = target_density*(1-proportion)
        the_sample = subsample_with_function(the_sample,constant_func_curry(subsampling_allowance))
        push!(sparsified_number_of_points,length(the_sample))
    end
    return (original_number_of_points,sparsified_number_of_points)
end

function naive_one_proportion(F,target_density,proportion)
    x = variables(F) 
    n = length(x)
    original_sample = sampling_fixed_density(F,target_density*proportion)
    subsampling_allowance = target_density*(1-proportion)
    the_sample = subsample_with_function(original_sample,constant_func_curry(subsampling_allowance))
    return (the_sample,original_sample)
end 

function feature_size_to_density(n,feature_size,delta)
    an = sqrt((2*n)/n+1)
    epsilon = (feature_size/2 - delta*an)/(2*(an^2 - 1/2))
    return epsilon
end

function homology_inference(F;wfs=nothing,subsampling_proportion=0.1,homology_up_to_degree=1,plot_diagrams=false,maximum_bottleneck_order=nothing,delta=1e-10,threshold=2e-4)
    if wfs == nothing
        wfs = compute_weak_feature_size(F;maximum_bottleneck_order=maximum_bottleneck_order,threshold=threshold)
    end
    ambient_dimension = length(variables(F))
    target = feature_size_to_density(ambient_dimension,wfs,delta)
    subsample, = naive_one_proportion(F,target,subsampling_proportion)

    # subsample is now a sample dense enough to perform homology inference
    starting_radius_for_homology_inference = 2*target 
    ending_radius_for_homology_inference = 2*(2*target*sqrt((2*ambient_dimension)/(ambient_dimension+1))+delta)
    maximum_radius_required = ending_radius_for_homology_inference*1.01
    persistence_diagrams = ripserer(subsample,threshold=maximum_radius_required,dim_max=homology_up_to_degree)    

    # Plot if desired
    if plot_diagrams
        plot(persistence_diagrams)
    end

    betti_numbers = []
    # Count the points contributing to each homology degree
    degree_index = 1
    for diagram in persistence_diagrams
        betti_number_points_in_this_diagram = [point for point in diagram if point[1] <= starting_radius_for_homology_inference && point[2] >= ending_radius_for_homology_inference]
        push!(betti_numbers,length(betti_number_points_in_this_diagram))
    end
    return betti_numbers
end
