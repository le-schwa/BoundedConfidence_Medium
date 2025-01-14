module BoundedConfidence_Medium

using Pkg
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("Graphs")
Pkg.add("GraphRecipes")
Pkg.add("StatsPlots")
Pkg.add("LaTeXStrings")
Pkg.add("StatsBase")
Pkg.add("ImageCore")

using Plots
using Random
using Distributions
using Graphs
using GraphRecipes
using StatsPlots
using LaTeXStrings
using StatsBase
using ImageCore

Random.seed!(123)

# Bounded Confidence Model
function bounded_confidence(N,ϵ_T,μ_T,T)
    t = 0
    x = rand(Uniform(0,1),N)
    y = x

    while t < T 
        x_1 = rand(collect(1:1:N))
        x_2 = rand(collect(1:1:N))

        if (abs(x[x_1]-x[x_2])<ϵ_T)
            x[x_1] = x[x_1]+μ_T*(x[x_2]-x[x_1])
            x[x_2] = x[x_2]+μ_T*(x[x_1]-x[x_2])
        end
        y = hcat(y,x)
        t = t+1 

    end
    p = plot(y',label=false)
    display(p)
    return y
end

# Examples
z = bounded_confidence(50,0.2,0.5,10000)
z = bounded_confidence(50,0.4,0.5,10000)
z = bounded_confidence(50,0.1,0.5,10000)


# Bounded Confidence Model with medium
function bc_medium(N,ϵ_T,μ_T,ϵ_A,μ_A,T)
    t = 0
    x = bounded_confidence(N,ϵ_T,μ_T,10^4)[:,2]
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:N
            x_1 = rand(collect(1:1:N+1))
            x_2 = rand(collect(1:1:N+1))
            if ((x_1 < N+1)&&(x_2 < N+1)) 
                if (abs(x[x_1]-x[x_2])<ϵ_T)
                    x[x_1]=x[x_1]+μ_T*(x[x_2]-x[x_1])
                    x[x_2]=x[x_2]+μ_T*(x[x_1]-x[x_2])
                end
            end
            if ((x_1 == N+1)&&(x_2 != N+1))
                if (abs(x[x_2]-A)<ϵ_A)
                    x[x_2]=x[x_2]+μ_A*(A-x[x_2])
                end
                if (abs(x[x_2]-A)>ϵ_A)
                    A = A+μ_A*(x[x_2]-A)
                end
            end
            if ((x_2 == N+1)&&(x_1 != N+1))
                if (abs(x[x_1]-A)<ϵ_A)
                    x[x_1]=x[x_1]+μ_A*(A-x[x_1])
                end
                if (abs(x[x_1]-A)>ϵ_A)
                    A=A+μ_A*(x[x_1]-A)
                end
            end

        end
        y = hcat(y,vcat(x,A))
        t = t+1
    end

    p = plot(y'[:,1:N-1],label=false,color = :blue)
    plot!(y'[:,N+1],label="A",linewidth=2)
    display(p)
    return y 
end

# Examples
m = bc_medium(100,0.2,0.5,0.1,0.05,2000)  
m = bc_medium(100,0.2,0.5,0.1,0.25,2000) 
m = bc_medium(100,0.2,0.5,0.1,0.6,2000)


# Bounded Confidence Model with medium and agent renewal 
function bc_medium_renewal(N,ϵ_T,μ_T,ϵ_A,μ_A,p_new,T)
    t = 0
    x = bounded_confidence(N,ϵ_T,μ_T,10000)[:,10000]
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:N
            x_1 = rand(collect(1:1:N+1))
            x_2 = rand(collect(1:1:N+1))
            if x_1 < N+1 & x_2 < N+1 
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    else 
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_2]-x[x_1])
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                end
            elseif x_1 == N+1 & x_2 != N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_2] = rand(Uniform(0,1))
                end
                if abs(x[x_2]-A) <= ϵ_A
                    x[x_2] = x[x_2] + μ_A*(A-x[x_2])
                elseif abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif x_2 == N+1 & x_1 != N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_1] = rand(Uniform(0,1))
                end
                if abs(x[x_1]-A) <= ϵ_A
                    x[x_1] = x[x_1] + μ_A*(A-x[x_1])
                elseif abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            end

        end
        y = hcat(y,vcat(x,A))
        t = t+1
    end

    p = plot(y'[:,1:N-1],label=false,color = :blue)
    plot!(y'[:,N+1],label="A",linewidth=2,color = :red,legend = :outertopright)
    display(p)
    return y 
end

# Examples 
m = bc_medium_renewal(30,0.2,0.5,0.6,0.1,0.03,1000)
m = bc_medium_renewal(30,0.2,0.5,0.45,0.1,0.03,1000)
m = bc_medium_renewal(30,0.2,0.5,0.2,0.1,0.03,1000)
m = bc_medium_renewal(30,0.2,0.5,0.1,0.1,0.03,1000)


# Bounded Confidence Model with medium, agent renewal and zealots
function bc_medium_renewal_zealots(N,ϵ_T,μ_T,ϵ_A,μ_A,p_new,Z0,Z1,T)
    t = 0
    x = bounded_confidence(N,ϵ_T,μ_T,10000)[:,10000]
    z = vcat(ones(Z1),zeros(Z0))
    x = vcat(x,z)
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:(N+Z0+Z1)
            x_1 = rand(collect(1:1:(N+Z0+Z1+1)))
            x_2 = rand(collect(1:1:(N+Z0+Z1+1)))
            if x_1 <= N && x_2 <= N                            # x_1 and x_2 are normal agents
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    else 
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_2]-x[x_1])
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                end
            elseif (N < x_1 <= (N+Z0+Z1)) && x_2 <= N        # x_1 is a zealot, x_2 is a normal agent
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                end
            elseif (N < x_2 <= (N+Z0+Z1)) && x_1 <= N        # x_2 is a zealot, x_1 is a normal agent
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_1]-x[x_2])
                end
            elseif (x_1 == (N+Z0+Z1+1)) && x_2 <= N                 # x_1 is the medium, x_2 is a normal agent
                if rand(Bernoulli(p_new)) == true
                    x[x_2] = rand(Uniform(0,1))
                end
                if abs(x[x_2]-A) <= ϵ_A
                    x[x_2] = x[x_2] + μ_A*(A-x[x_2])
                elseif abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif (x_2 == (N+Z0+Z1+1)) && x_1 <= N                 # x_2 is the medium, x_1 is a normal agent
                if rand(Bernoulli(p_new)) == true
                    x[x_1] = rand(Uniform(0,1))
                end
                if abs(x[x_1]-A) <= ϵ_A
                    x[x_1] = x[x_1] + μ_A*(A-x[x_1])
                elseif abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            elseif (x_1 == (N+Z0+Z1+1)) && (N < x_2 <= (N+Z0+Z1))       # x_1 is the medium, x_2 is a zealot
                if abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif (x_2 == (N+Z0+Z1+1)) && (N < x_2 <= (N+Z0+Z1))       # x_2 is the medium, x_1 is a zealot
                if abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            end

        end
        y = hcat(y,vcat(x,A))
        t = t+1
    end

    p = plot(y'[:,1:N],label=false,color = :blue)
    plot!(y'[:,N+1],label="Zealot",linewidth = 2,color = :purple)
    plot!(y'[:,N+2:N+Z0+Z1],linewidth = 2,color = :purple,label=false)
    plot!(y'[:,N+Z0+Z1+1],label="A",linewidth=2,color = :red,legend = :outertopright)
    # title!(L"\varepsilon_A = %$ϵ_A \ \mu_A = %$μ_A" )
    display(p)
    return y 
end

# Examples
m = bc_medium_renewal_zealots(30,0.2,0.5,0.1,0.05,0.0,1,1,2000)  
m = bc_medium_renewal_zealots(100,0.2,0.5,0.42,0.1,0.01,1,1,2000)
# How to compute different statistical variables for the above example
m[103,:]
histogram(m[103,:])
StatsPlots.density(m[103,:])
mean(m[103,:])
var(m[103,:])
mode(m[103,:])


# Examples for different graphs
g = erdos_renyi(30,0.5)
g = clique_graph(30,5)
g = barabasi_albert(30,20)
add_vertex!(g)
for i in 1:(nv(g)-1)
    add_edge!(g,i,nv(g))
end
# Visualization of the graph
g_c = ["blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","blue","red"]
graphplot(g, curves=false,nodeshape = :circle, method = :stress, markercolor = g_c,markersize = 0.2)

# Function to assign same color for agents with similar opinions
f = colorsigned(colorant"red",colorant"green",colorant"blue") ∘ scalesigned(0.0, 0.5, 1.0)

# Bounded Confidence Model with medium, agent renewal and erdös-renyi graph
function bc_medium_renewal_erdös(N,p,ϵ_T,μ_T,ϵ_A,μ_A,p_new,T)
    t = 0
    g = erdos_renyi(N,p)
    add_vertex!(g)
    for i in 1:(nv(g)-1)
        add_edge!(g,i,nv(g))
    end
    gp = graphplot(g, curves=false,nodeshape = :circle, method = :stress, markercolor = "red")
    display(gp)
    x = bounded_confidence(N,ϵ_T,μ_T,10000)[:,10000]
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:N
            x_1 = rand(collect(1:1:N+1))
            n = Graphs.neighbors(g,x_1)
            x_2 = n[rand(collect(1:1:length(n)))]
            if x_1 < N+1 && x_2 < N+1 
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    else 
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_2]-x[x_1])
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                end
            elseif x_1 == N+1 && x_2 < N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_2] = rand(Uniform(0,1))
                end
                if abs(x[x_2]-A) <= ϵ_A
                    x[x_2] = x[x_2] + μ_A*(A-x[x_2])
                elseif abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif x_2 == N+1 && x_1 < N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_1] = rand(Uniform(0,1))
                end
                if abs(x[x_1]-A) <= ϵ_A
                    x[x_1] = x[x_1] + μ_A*(A-x[x_1])
                elseif abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            end

        end
        y = hcat(y,vcat(x,A))
        t = t+1
    end

    p = plot(y'[:,1:N-1],label=false,color = :blue)
    plot!(y'[:,N+1],label="A",linewidth=2,color = :red)
    display(p)
    return [y,g] 
end

# Examples
m = bc_medium_renewal_erdös(200,0.85,0.2,0.5,0.45,0.1,0.005,5000)
m = bc_medium_renewal_erdös(30,0.75,0.2,0.5,0.45,0.1,0.03,1000)
# Plots the underlying network with coloring for the vertices
graphplot(m[2], curves=false,nodeshape = :circle, method = :stress, markercolor = f.(m[1][:,size(m[1])[2]]),markersize = 0.2)

# Bounded Confidence Model with medium, agent renewal and barabasi albert graph
function bc_medium_renewal_ba(N,k,ϵ_T,μ_T,ϵ_A,μ_A,p_new,T)
    t = 0
    g = barabasi_albert(N,k)
    add_vertex!(g)
    for i in 1:(nv(g)-1)
        add_edge!(g,i,nv(g))
    end
    gp = graphplot(g, curves=false,nodeshape = :circle, method = :stress, markercolor = "red")
    display(gp)
    x = bounded_confidence(N,ϵ_T,μ_T,10000)[:,2]
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:N
            x_1 = rand(collect(1:1:N+1))
            n = Graphs.neighbors(g,x_1)
            x_2 = n[rand(collect(1:1:length(n)))]
            if x_1 < N+1 && x_2 < N+1 
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    else 
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_2]-x[x_1])
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                end
            elseif x_1 == N+1 && x_2 < N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_2] = rand(Uniform(0,1))
                end
                if abs(x[x_2]-A) <= ϵ_A
                    x[x_2] = x[x_2] + μ_A*(A-x[x_2])
                elseif abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif x_2 == N+1 && x_1 < N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_1] = rand(Uniform(0,1))
                end
                if abs(x[x_1]-A) <= ϵ_A
                    x[x_1] = x[x_1] + μ_A*(A-x[x_1])
                elseif abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            end

        end
        y = hcat(y,vcat(x,A))
        t = t+1
    end

    p = plot(y'[:,1:N-1],label=false,color = :blue)
    plot!(y'[:,N+1],label="A",linewidth=2,color = :red)
    display(p)
    return [y,g] 
end

# Example
m = bc_medium_renewal_ba(100,45,0.2,0.5,0.42,0.1,0.01,2000)
graphplot(m[2], curves=false,nodeshape = :circle, method = :stress, markercolor = f.(m[1][:,size(m[1])[2]]),markersize = 0.2)

# Bounded Confidence Model with medium, agent renewal and clique graph
function bc_medium_renewal_clique(N,k,ϵ_T,μ_T,ϵ_A,μ_A,p_new,T)
    t = 0
    g = clique_graph(Int.(N/k),k)
    add_vertex!(g)
    for i in 1:(nv(g)-1)
        add_edge!(g,i,nv(g))
    end
    gp = graphplot(g, curves=false,nodeshape = :circle, method = :stress, markercolor = "red")
    display(gp)
    x = bounded_confidence(N,ϵ_T,μ_T,10000)[:,10000]
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:N
            x_1 = rand(collect(1:1:N+1))
            n = Graphs.neighbors(g,x_1)
            x_2 = n[rand(collect(1:1:length(n)))]
            if x_1 < N+1 && x_2 < N+1 
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    else 
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_2]-x[x_1])
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                end
            elseif x_1 == N+1 && x_2 < N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_2] = rand(Uniform(0,1))
                end
                if abs(x[x_2]-A) <= ϵ_A
                    x[x_2] = x[x_2] + μ_A*(A-x[x_2])
                elseif abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif x_2 == N+1 && x_1 < N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_1] = rand(Uniform(0,1))
                end
                if abs(x[x_1]-A) <= ϵ_A
                    x[x_1] = x[x_1] + μ_A*(A-x[x_1])
                elseif abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            end

        end
        y = hcat(y,vcat(x,A))
        t = t+1
    end

    p = plot(y'[:,1:N-1],label=false,color = :blue)
    plot!(y'[:,N+1],label="A",linewidth=2,color = :red)
    display(p)
    return [y,g] 
end

# Example
m = bc_medium_renewal_clique(100,25,0.2,0.5,0.42,0.1,0.01,2000)
graphplot(m[2], curves=false,nodeshape = :circle, method = :stress, markercolor = f.(m[1][:,size(m[1])[2]]),markersize = 0.2)






# Possible further extension with negative feedback
# Was not covered in the essay
function bc_medium_renewal_negative_feedback(N,ϵ_T,μ_T,ϵ_A,μ_A,p_new,T)
    t = 0
    x = bounded_confidence(N,ϵ_T,μ_T,10000)[:,10000]
    A = rand(Uniform(0,1))
    y = vcat(x,A)

    while t < T 
        for i in 1:N
            x_1 = rand(collect(1:1:N+1))
            x_2 = rand(collect(1:1:N+1))
            if x_1 < N+1 && x_2 < N+1 
                if rand(Bernoulli(p_new)) == true
                    if rand(Bernoulli(0.5)) == true
                        x[x_1] = rand(Uniform(0,1))
                    else 
                        x[x_2] = rand(Uniform(0,1))
                    end
                end
                if abs(x[x_1]-x[x_2]) < ϵ_T
                    x[x_1] = x[x_1] + μ_T*(x[x_2]-x[x_1])
                    x[x_2] = x[x_2] + μ_T*(x[x_1]-x[x_2])
                else
                    x[x_1] = (x[x_1] - μ_T*(x[x_2]-x[x_1]))*(x[x_1] - μ_T*(x[x_2]-x[x_1]) >= 0 && x[x_1] - μ_T*(x[x_2]-x[x_1]) <= 1)
                    x[x_2] = (x[x_2] - μ_T*(x[x_1]-x[x_2]))*(x[x_2] - μ_T*(x[x_1]-x[x_2]) >= 0 && x[x_2] - μ_T*(x[x_1]-x[x_2]) <= 1)
                end
            elseif x_1 == N+1 && x_2 != N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_2] = rand(Uniform(0,1))
                end
                if abs(x[x_2]-A) <= ϵ_A
                    x[x_2] = x[x_2] + μ_A*(A-x[x_2])
                elseif abs(x[x_2]-A) > ϵ_A
                    A = A + μ_A*(x[x_2]-A)
                end
            elseif x_2 == N+1 && x_1 != N+1
                if rand(Bernoulli(p_new)) == true
                    x[x_1] = rand(Uniform(0,1))
                end
                if abs(x[x_1]-A) <= ϵ_A
                    x[x_1] = x[x_1] + μ_A*(A-x[x_1])
                elseif abs(x[x_1]-A) > ϵ_A
                    A = A + μ_A*(x[x_1]-A)
                end
            end

            y = hcat(y,vcat(x,A))
        end

        t = t+1
    end

    p = plot(y'[:,1:N-1],label=false,color = :blue)
    plot!(y'[:,N+1],label="A",linewidth=2,color = :red)
    display(p)
    return y 
end

# Example
bc_medium_renewal_negative_feedback(30,0.2,0.5,0.1,0.1,0,1000)

end # module BoundedConfidence_Medium
