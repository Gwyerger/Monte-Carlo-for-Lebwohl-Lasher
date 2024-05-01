using Plots
using PlotThemes
theme(:wong)

using LinearAlgebra
using Statistics
using StatsPlots
using KernelDensity
using Unitful
using Measures
using Dierckx

function wrap_index(i::Int, l::Int)::Int
	wrap = (i - 1) % l + 1
	return (wrap <= 0) ? l + wrap : wrap
end

function RandAnglePhi()
	i=2*rand()-1
	return i*pi
end

function Random3Vector()
	i = 2*rand()-1
    j = 2*rand()-1
    k = rand()
    n = [i,j,k] ./ norm([i,j,k])
	return n
end

function wrap_angle(θ)
    while θ <= -pi/2
        θ += pi 
    end
    while θ > pi/2
        θ -= pi
    end
    return θ
end

mutable struct Mesh 
    Nx::Int
    Ny::Int
    dx::Float64
    dy::Float64
    N::Int
    MeshGridCoords::Array
end

mutable struct LebwohlLasher2D
    Nx::Int
    Ny::Int
    T::Float64
    ϵ::Float64
    accepted_moves::Int
    state::Array
    grid::Mesh
end

function Mesh(Nx::Int, Ny::Int = 0)
    if Ny == 0
        Ny = Nx
    end 
    dx = 1
    dy = 1
    return Mesh(Nx, Ny, dx, dy, Nx*Ny, Array(stack([[[x*dx, y*dy] for x in 1:Nx] for y in 1:Ny])))
end

function LebwohlLasher2D(grid::Mesh, T::Float64, ϵ::Float64, IC::String="Uniform")
    Nx = grid.Nx
    Ny = grid.Ny
    if IC=="Uniform"
        state = zeros(Float64, Nx, Ny)
    end
    if IC=="Random"
        state = rand(Nx, Ny).*pi
    end
    return LebwohlLasher2D(Nx, Ny, T, ϵ, 0, state, grid)
end

function PairwiseEnergy2D(state1::Float64, state2::Float64, ϵ::Float64) 
    θ1 = state1
    θ2 = state2
    cosΔθ = cos(θ1 - θ2)
    P2 = 1/2*(3*cosΔθ^2 - 3)
    E = -ϵ*P2
end


function PairwiseEnergy2D_BendPenalty(state1::Float64, state2::Float64, ϵ::Float64; β::Float64 = π/4)
    # This is a Bend Modulated Energy
    θ1 = state1
    θ2 = state2
    cosΔθ = cos(abs(θ1 - θ2) - β)
    P2 = 1/2*(3*cosΔθ^2 - 3)
    E = -ϵ*P2
end

function LocalEnergy2D(locale::Array, ϵ::Float64, periodic::Bool, VEdge, HEdge; InteractionPotential = PairwiseEnergy2D)
    if periodic == true || (VEdge == nothing && HEdge == nothing)
        center = [2, 2]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[1, 2], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[2, 1], ϵ)
        e3 = InteractionPotential(locale[center[1], center[2]], locale[3, 2], ϵ)
        e4 = InteractionPotential(locale[center[1], center[2]], locale[2, 3], ϵ)
        LocalEnergy = e1 + e2 + e3 + e4
    elseif VEdge == "Top" && HEdge == nothing
        center = [2, 1]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[1, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[3, 1], ϵ)
        e3 = InteractionPotential(locale[center[1], center[2]], locale[2, 2], ϵ)
        LocalEnergy = e1 + e2 + e3
    elseif VEdge == "Bottom" && HEdge == nothing
        center = [2, 2]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[1, 2], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[3, 2], ϵ)
        e3 = InteractionPotential(locale[center[1], center[2]], locale[2, 1], ϵ)
        LocalEnergy = e1 + e2 + e3
    elseif VEdge == nothing && HEdge == "Left"
        center = [1, 2]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[1, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[1, 3], ϵ)
        e3 = InteractionPotential(locale[center[1], center[2]], locale[2, 2], ϵ)
        LocalEnergy = e1 + e2 + e3
    elseif VEdge == nothing && HEdge == "Right"
        center = [2, 2]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[2, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[2, 3], ϵ)
        e3 = InteractionPotential(locale[center[1], center[2]], locale[1, 2], ϵ)
        LocalEnergy = e1 + e2 + e3
    elseif VEdge == "Top" && HEdge == "Left"
        center = [1, 1]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[2, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[1, 2], ϵ)
        LocalEnergy = e1 + e2
    elseif VEdge == "Top" && HEdge == "Right"
        center = [2, 1]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[1, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[2, 2], ϵ)
        LocalEnergy = e1 + e2
    elseif VEdge == "Bottom" && HEdge == "Left"
        center = [1, 2]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[1, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[2, 2], ϵ)
        LocalEnergy = e1 + e2
    elseif VEdge == "Bottom" && HEdge == "Right"
        center = [2, 2]
        e1 = InteractionPotential(locale[center[1], center[2]], locale[2, 1], ϵ)
        e2 = InteractionPotential(locale[center[1], center[2]], locale[1, 2], ϵ)
        LocalEnergy = e1 + e2
    end
    return LocalEnergy
end

function MCStepper2D!(sys::LebwohlLasher2D, iterations::Int, maxStepSize::Float64, BC::String; BoundaryOnly::Bool = false, fix::BitMatrix = BitMatrix(zeros(Bool, sys.Nx, sys.Ny)), InteractionPotential = PairwiseEnergy2D)
    Nx = sys.Nx
    Ny = sys.Ny
    ϵ = sys.ϵ
    T = sys.T
    Padding = 2
    if BC == "Periodic"
        periodic = true
        VStart = 1
        HStart = 1
        VEnd = Ny
        HEnd = Nx
    elseif BC == "Free"
        periodic = false
        VStart = 1
        HStart = 1
        VEnd = Ny
        HEnd = Nx
    elseif BC == "Fixed"
        periodic = false
        VStart = 2
        HStart = 2
        VEnd = Ny-1
        HEnd = Nx-1
    end
    for h in 1:iterations
        for ii in HStart:HEnd
            HEdge = nothing
            if ii==1
                HEdge = "Left"
            elseif ii==Nx
                HEdge = "Right"
            end
            for jj in VStart:VEnd
                VEdge = nothing
                if jj==1
                    VEdge = "Top"
                elseif jj==Ny
                    VEdge = "Bottom"
                end
                if BoundaryOnly
                    if !(ii <= HStart + Padding) && !(ii >= HEnd - Padding) && !(jj <= VStart + Padding) && !(jj >= VEnd - Padding)
                        continue
                    end
                end
                if fix[ii, jj] == true
                    continue
                end
                if periodic == true
                    indexi = [wrap_index(ii-1, Nx), ii, wrap_index(ii+1, Nx)]
                    indexj = [wrap_index(jj-1, Ny), jj, wrap_index(jj+1, Ny)]
                elseif VEdge == nothing && HEdge == nothing
                    indexi = [ii-1, ii, ii+1]
                    indexj = [jj-1, jj, jj+1]
                elseif VEdge == "Top" && HEdge == nothing
                    indexi = [ii-1, ii, ii+1]
                    indexj = [jj, jj+1]
                elseif VEdge == "Bottom" && HEdge == nothing
                    indexi = [ii-1, ii, ii+1]
                    indexj = [jj-1, jj]
                elseif VEdge == nothing && HEdge == "Left"
                    indexi = [ii, ii+1]
                    indexj = [jj-1, jj, jj+1]
                elseif VEdge == nothing && HEdge == "Right"
                    indexi = [ii-1, ii]
                    indexj = [jj-1, jj, jj+1] 
                elseif VEdge == "Top" && HEdge == "Left"
                    indexi = [ii, ii+1]
                    indexj = [jj, jj+1]
                elseif VEdge == "Top" && HEdge == "Right"
                    indexi = [ii-1, ii]
                    indexj = [jj, jj+1]
                elseif VEdge == "Bottom" && HEdge == "Left"
                    indexi = [ii, ii+1]
                    indexj = [jj-1, jj]
                elseif VEdge == "Bottom" && HEdge == "Right"
                    indexi = [ii-1, ii]
                    indexj = [jj-1, jj]
                end
                locale = Float64[sys.state[iii, jjj] for iii in indexi, jjj in indexj]
                Ei = LocalEnergy2D(locale, ϵ, periodic, VEdge, HEdge, InteractionPotential=InteractionPotential)
                localenew = locale
                rstep = RandAnglePhi()*maxStepSize
                localenew[2,2] = wrap_angle(localenew[2,2] + rstep)
                Ef = LocalEnergy2D(localenew, ϵ, periodic, VEdge, HEdge, InteractionPotential=InteractionPotential)
                w = exp((Ei-Ef)/T)
                wr = rand()
                if wr < w
                    sys.state[ii, jj] = localenew[2,2]
                    sys.accepted_moves +=1
                end
            end
        end
    end
end

function FreeEnergy2D(sys::LebwohlLasher2D, BC::String; InteractionPotential = PairwiseEnergy2D)
    Nx = sys.Nx
    Ny = sys.Ny
    ϵ = sys.ϵ
    TotalEnergy = 0
    if BC == "Periodic"
        periodic = true
        VStart = 1
        HStart = 1
        VEnd = Ny
        HEnd = Nx
    elseif BC == "Free"
        periodic = false
        VStart = 1
        HStart = 1
        VEnd = Ny
        HEnd = Nx
    elseif BC == "Fixed"
        periodic = false
        VStart = 2
        HStart = 2
        VEnd = Ny-1
        HEnd = Nx-1
    end
    for ii in HStart:HEnd
        HEdge = nothing
        if ii==1
            HEdge = "Left"
        elseif ii==Nx
            HEdge = "Right"
        end
        for jj in VStart:VEnd
            VEdge = nothing
            if jj==1
                VEdge = "Top"
            elseif jj==Ny
                VEdge = "Bottom"
            end
            if periodic == true
                indexi = [wrap_index(ii-1, Nx), ii, wrap_index(ii+1, Nx)]
                indexj = [wrap_index(jj-1, Ny), jj, wrap_index(jj+1, Ny)]
            elseif VEdge == nothing && HEdge == nothing
                indexi = [ii-1, ii, ii+1]
                indexj = [jj-1, jj, jj+1]
            elseif VEdge == "Top" && HEdge == nothing
                indexi = [ii-1, ii, ii+1]
                indexj = [jj, jj+1]
            elseif VEdge == "Bottom" && HEdge == nothing
                indexi = [ii-1, ii, ii+1]
                indexj = [jj-1, jj]
            elseif VEdge == nothing && HEdge == "Left"
                indexi = [ii, ii+1]
                indexj = [jj-1, jj, jj+1]
            elseif VEdge == nothing && HEdge == "Right"
                indexi = [ii-1, ii]
                indexj = [jj-1, jj, jj+1] 
            elseif VEdge == "Top" && HEdge == "Left"
                indexi = [ii, ii+1]
                indexj = [jj, jj+1]
            elseif VEdge == "Top" && HEdge == "Right"
                indexi = [ii-1, ii]
                indexj = [jj, jj+1]
            elseif VEdge == "Bottom" && HEdge == "Left"
                indexi = [ii, ii+1]
                indexj = [jj-1, jj]
            elseif VEdge == "Bottom" && HEdge == "Right"
                indexi = [ii-1, ii]
                indexj = [jj-1, jj]
            end
            locale = Float64[sys.state[iii, jjj] for iii in indexi, jjj in indexj]
            Eij = LocalEnergy2D(locale, ϵ, periodic, VEdge, HEdge, InteractionPotential=InteractionPotential)
            TotalEnergy += Eij
        end
    end
    TotalEnergy = TotalEnergy/2
    # println("Total Free Energy: "*string(TotalEnergy))
    return TotalEnergy
end

function EnergyDensity2D(sys::LebwohlLasher2D, BC::String; InteractionPotential = PairwiseEnergy2D)
    Nx = sys.Nx
    Ny = sys.Ny
    ϵ = sys.ϵ
    f = zeros(Float64, Nx, Ny)
    if BC == "Periodic"
        periodic = true
        VStart = 1
        HStart = 1
        VEnd = Ny
        HEnd = Nx
    elseif BC == "Free"
        periodic = false
        VStart = 1
        HStart = 1
        VEnd = Ny
        HEnd = Nx
    elseif BC == "Fixed"
        periodic = false
        VStart = 2
        HStart = 2
        VEnd = Ny-1
        HEnd = Nx-1
    end
    for ii in HStart:HEnd
        HEdge = nothing
        if ii==1
            HEdge = "Left"
        elseif ii==Nx
            HEdge = "Right"
        end
        for jj in VStart:VEnd
            VEdge = nothing
            if jj==1
                VEdge = "Top"
            elseif jj==Ny
                VEdge = "Bottom"
            end
            if periodic == true
                indexi = [wrap_index(ii-1, Nx), ii, wrap_index(ii+1, Nx)]
                indexj = [wrap_index(jj-1, Ny), jj, wrap_index(jj+1, Ny)]
            elseif VEdge == nothing && HEdge == nothing
                indexi = [ii-1, ii, ii+1]
                indexj = [jj-1, jj, jj+1]
            elseif VEdge == "Top" && HEdge == nothing
                indexi = [ii-1, ii, ii+1]
                indexj = [jj, jj+1]
            elseif VEdge == "Bottom" && HEdge == nothing
                indexi = [ii-1, ii, ii+1]
                indexj = [jj-1, jj]
            elseif VEdge == nothing && HEdge == "Left"
                indexi = [ii, ii+1]
                indexj = [jj-1, jj, jj+1]
            elseif VEdge == nothing && HEdge == "Right"
                indexi = [ii-1, ii]
                indexj = [jj-1, jj, jj+1] 
            elseif VEdge == "Top" && HEdge == "Left"
                indexi = [ii, ii+1]
                indexj = [jj, jj+1]
            elseif VEdge == "Top" && HEdge == "Right"
                indexi = [ii-1, ii]
                indexj = [jj, jj+1]
            elseif VEdge == "Bottom" && HEdge == "Left"
                indexi = [ii, ii+1]
                indexj = [jj-1, jj]
            elseif VEdge == "Bottom" && HEdge == "Right"
                indexi = [ii-1, ii]
                indexj = [jj-1, jj]
            end
            locale = Float64[sys.state[iii, jjj] for iii in indexi, jjj in indexj]
            f[ii,jj] = LocalEnergy2D(locale, ϵ, periodic, VEdge, HEdge, InteractionPotential = InteractionPotential)
        end
    end
    return f
end


function QtensorPoints2D(sys::LebwohlLasher2D; rotation::Float64 = 0.0)
    rotatedstate = wrap_angle.(sys.state .+ rotation)
    Qxx =  cos.(rotatedstate[:,:]) .^2 .- 1/2
    Qxy =  cos.(rotatedstate[:,:]) .*sin.(rotatedstate[:,:])
    return Qxx, Qxy
end

function Qtensor2D(sys::LebwohlLasher2D)
    Qxx = mean(reduce(vcat, cos.(sys.state[:,:]) .^2 .- 1/2))
    Qxy = mean(reduce(vcat, cos.(sys.state[:,:]) .*sin.(sys.state[:,:])))
    return Qxx, Qxy
end


mutable struct LebwohlLasher3D
    N::Int
    T::Float64
    ϵ::Float64
    accepted_moves::Int
    state::Array
end

function LebwohlLasher3D(N::Int, T::Float64, ϵ::Float64, IC::String="Uniform")
    if IC=="Uniform"
        state = zeros(Float64, N, N, N, 3)
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    state[i,j,k,:] = [0,0,1]
                end
            end
        end
    elseif IC=="Random"
        state = zeros(Float64, N, N, N, 3)
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    state[i,j,k,:] = Random3Vector()
                end
            end
        end
    end
    return LebwohlLasher3D(N, T, ϵ, 0, state)
end

function PairwiseEnergy3D(state1::Array, state2::Array, ϵ::Float64) 
    state1
    state2
    P2 = 1/2*(3*(dot(state1,state2))^2 - 3)
    E = -ϵ*P2
end

function LocalEnergy3D(locale::Array, ϵ::Float64)
    if size(locale)==(3, 3, 3, 3)
        e1 = PairwiseEnergy3D(locale[2, 2, 2, :], locale[1, 2, 2, :], ϵ)
        e2 = PairwiseEnergy3D(locale[2, 2, 2, :], locale[2, 1, 2, :], ϵ)
        e3 = PairwiseEnergy3D(locale[2, 2, 2, :], locale[2, 2, 1, :], ϵ)
        e4 = PairwiseEnergy3D(locale[2, 2, 2, :], locale[3, 2, 2, :], ϵ)
        e5 = PairwiseEnergy3D(locale[2, 2, 2, :], locale[2, 3, 2, :], ϵ)
        e6 = PairwiseEnergy3D(locale[2, 2, 2, :], locale[2, 2, 3, :], ϵ)
        LocalEnergy = e1 + e2 + e3 + e4 + e5 + e6
    else print("Improperly Defined Locale")
    end
    return LocalEnergy
end


function MCStepper3D!(sys::LebwohlLasher3D, iterations::Int, maxStepSize::Float64, BC::String)
    N = sys.N
    ϵ = sys.ϵ
    T = sys.T
    if BC == "Periodic"
        VStart = 1
        HStart = 1
        KStart = 1
        VEnd = N
        HEnd = N
        KEnd = N
    elseif BC == "Fixed"
        VStart = 2
        HStart = 2
        KStart = 2
        VEnd = N-1
        HEnd = N-1
        KEnd = N-1
    end
    for h in 1:iterations
        for i in HStart:HEnd
            for j in VStart:VEnd
                for k in KStart:KEnd
                    indexi = [wrap_index(i-1, N), i, wrap_index(i+1, N)]
                    indexj = [wrap_index(j-1, N), j, wrap_index(j+1, N)]
                    indexk = [wrap_index(k-1, N), k, wrap_index(k+1, N)]
                    locale = Float64[sys.state[ii, jj, kk, ll] for ii in indexi, jj in indexj, kk in indexk, ll in 1:3]
                    Ei = LocalEnergy3D(locale, ϵ)
                    localenew = locale
                    dnew = Random3Vector()
                    localenew[2,2,2, :] = dnew
                    Ef = LocalEnergy3D(localenew, ϵ)
                    w = exp((Ei-Ef)/T)
                    if rand() < w
                        sys.state[i, j, k, :] = localenew[2,2,2, :]
                        sys.accepted_moves +=1
                    end
                end
            end
        end
    end
end

function FreeEnergy3D(sys::LebwohlLasher3D)
    N = sys.N
    ϵ = sys.ϵ
    TotalEnergy = 0
    for i in 1:N
        for j in 1:N
            for k in 1:N
                indexi = [wrap_index(i-1, N), i, wrap_index(i+1, N)]
                indexj = [wrap_index(j-1, N), j, wrap_index(j+1, N)]
                indexk = [wrap_index(k-1, N), k, wrap_index(k+1, N)]
                locale = Float64[sys.state[ii, jj, kk, ll] for ii in indexi, jj in indexj, kk in indexk, ll in 1:3]
                TotalEnergy += LocalEnergy3D(locale, ϵ)
            end
        end
    end
    TotalEnergy = TotalEnergy/2
    println("Total Free Energy: "*string(TotalEnergy))
    return TotalEnergy
end

function EnergyDensity3D(sys::LebwohlLasher3D, BC::String)
    Nx = sys.N
    Ny = sys.N
    Nz = sys.N
    ϵ = sys.ϵ
    f = zeros(Float64, Nx, Ny, Nz)
    if BC == "Periodic"
        periodic = true
        VStart = 1
        HStart = 1
        ZStart = 1
        VEnd = Ny
        HEnd = Nx
        ZEnd = Nz
    end
    for ii in HStart:HEnd
        for jj in VStart:VEnd
            for kk in ZStart:ZEnd
                if periodic == true
                    indexi = [wrap_index(ii-1, Nx), ii, wrap_index(ii+1, Nx)]
                    indexj = [wrap_index(jj-1, Ny), jj, wrap_index(jj+1, Ny)]
                    indexk = [wrap_index(kk-1, Nz), kk, wrap_index(kk+1, Nz)]
                end
                locale = Float64[sys.state[iii, jjj, kkk, ll] for iii in indexi, jjj in indexj, kkk in indexk, ll in 1:3]
                f[ii,jj,kk] = LocalEnergy3D(locale, ϵ)
            end
        end
    end
    return f
end

function QtensorPoints3D(sys::LebwohlLasher3D)
    Qxx = 3/2 .* (sys.state[:,:,:,1]) .^2 .- 1/2
    Qyy = 3/2 .* (sys.state[:,:,:,2]) .^2 .- 1/2
    Qzz = 3/2 .* (sys.state[:,:,:,3]) .^2 .- 1/2
    Qxy = 3/2 .* (sys.state[:,:,:,1]).*(sys.state[:,:,:,2])
    Qyz = 3/2 .* (sys.state[:,:,:,2]).*(sys.state[:,:,:,3])
    Qzx = 3/2 .* (sys.state[:,:,:,3]).*(sys.state[:,:,:,1])
    return Qxx, Qyy, Qzz, Qxy, Qyz, Qzx
end

function Qtensor3D(sys::LebwohlLasher3D)
    Qxx = mean(reduce(vcat,3/2 .* (sys.state[:,:,:,1]) .^2 .- 1/2))
    Qyy = mean(reduce(vcat,3/2 .* (sys.state[:,:,:,2]) .^2 .- 1/2))
    Qzz = mean(reduce(vcat,3/2 .* (sys.state[:,:,:,3]) .^2 .- 1/2))
    Qxy = mean(reduce(vcat,3/2 .* (sys.state[:,:,:,1]).*(sys.state[:,:,:,2])))
    Qyz = mean(reduce(vcat,3/2 .* (sys.state[:,:,:,2]).*(sys.state[:,:,:,3])))
    Qzx = mean(reduce(vcat,3/2 .* (sys.state[:,:,:,3]).*(sys.state[:,:,:,1])))
    return Qxx, Qyy, Qzz, Qxy, Qyz, Qzx
end

function VectorField2D(sys::LebwohlLasher2D)
    x = cos.(sys.state[:, :])
    y = sin.(sys.state[:, :])
    return x, y
end

function Director2D(sys::LebwohlLasher2D)
    Qxx, Qxy = Qtensor2D(sys)
    n = [(Qxx + sqrt(Qxx^2 + Qxy^2))/(Qxy*sqrt(1 + ((Qxx + sqrt(Qxx^2 + Qxy^2))/Qxy)^2)), 1/sqrt(1 + ((Qxx + sqrt(Qxx^2 + Qxy^2))/Qxy)^2)]
    return n
end 

function DirectorPlot2D(sys::LebwohlLasher2D; arrows::Bool = false, directional::Bool = false, biaxial::Bool=false)
    nx, ny = VectorField2D(sys)
    Nx = sys.Nx
    Ny = sys.Ny
    x = reduce(vcat, [[i for i in 1:Nx] for j in 1:Ny])
    y = reduce(vcat, [[j for i in 1:Nx] for j in 1:Ny])
    nx = reduce(vcat, nx)
    ny = reduce(vcat, ny)

    xl1 = x .- nx/2.5
    xl2 = x .+ nx/2.5
    yl1 = y .- ny/2.5
    yl2 = y .+ ny/2.5

    xb1 = x .+ ny/2.5
    xb2 = x .- ny/2.5
    yb1 = y .- nx/2.5
    yb2 = y .+ nx/2.5

    p = plot(dpi=300)
    if !arrows
        for i in 1:Nx*Ny
            plot!([xl1[i], xl2[i]],[yl1[i], yl2[i]], label = nothing, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            if biaxial
                plot!([xb1[i], xb2[i]],[yb1[i], yb2[i]], label = nothing, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            end
        end
    elseif arrows
        if !directional
            for i in 1:Nx*Ny
                plot!([x[i], xl2[i]],[y[i], yl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
                plot!([x[i], xl1[i]],[y[i], yl1[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            end
        elseif directional
            for i in 1:Nx*Ny
                plot!([xl1[i], xl2[i]],[yl1[i], yl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            end
        end
    end
    return p
end

function DirectorPlot2D!(sys::LebwohlLasher2D; arrows::Bool = false, directional::Bool = false, biaxial::Bool=false)
    nx, ny = VectorField2D(sys)
    Nx = sys.Nx
    Ny = sys.Ny
    x = reduce(vcat, [[i for i in 1:Nx] for j in 1:Ny])
    y = reduce(vcat, [[j for i in 1:Nx] for j in 1:Ny])
    nx = reduce(vcat, nx)
    ny = reduce(vcat, ny)

    xl1 = x .- nx/2.5
    xl2 = x .+ nx/2.5
    yl1 = y .- ny/2.5
    yl2 = y .+ ny/2.5
if !arrows
    for i in 1:Nx*Ny
        plot!([xl1[i], xl2[i]],[yl1[i], yl2[i]], label = nothing, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
        if biaxial
            plot!([xb1[i], xb2[i]],[yb1[i], yb2[i]], label = nothing, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
        end
    end
elseif arrows
    if !directional
        for i in 1:Nx*Ny
            plot!([x[i], xl2[i]],[y[i], yl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            plot!([x[i], xl1[i]],[y[i], yl1[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
        end
    elseif directional
        for i in 1:Nx*Ny
            plot!([xl1[i], xl2[i]],[yl1[i], yl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
        end
    end
end
end

function DirectorPlot3D(sys::LebwohlLasher3D; arrows::Bool = false, directional::Bool = false)
    nx = sys.state[:,:,:,1]
    ny = sys.state[:,:,:,2]
    nz = sys.state[:,:,:,3]
    Nx = sys.N
    Ny = sys.N
    Nz = sys.N
    x = reduce(vcat,reduce(vcat,[[[i for i in 1:Nx] for j in 1:Ny] for k in 1:Nz]))
    y = reduce(vcat,reduce(vcat,[[[j for i in 1:Nx] for j in 1:Ny] for k in 1:Nz]))
    z = reduce(vcat,reduce(vcat,[[[k for i in 1:Nx] for j in 1:Ny] for k in 1:Nz]))

    nx = reduce(vcat,reduce(vcat, nx))
    ny = reduce(vcat,reduce(vcat, ny))
    nz = reduce(vcat,reduce(vcat, nz))

    xl1 = x .- nx/2.5
    xl2 = x .+ nx/2.5
    yl1 = y .- ny/2.5
    yl2 = y .+ ny/2.5
    zl1 = z .- nz/2.5
    zl2 = z .+ nz/2.5

    p = plot3d(dpi=300)
    if !arrows
        for i in 1:Nx*Ny*Nz
            plot3d!([xl1[i], xl2[i]],[yl1[i], yl2[i]],[zl1[i], zl2[i]], label = nothing, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
        end
    elseif arrows
        if !directional
            for i in 1:Nx*Ny*Nz
                plot3d!([x[i], xl2[i]],[y[i], yl2[i]],[zl1[i], zl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
                plot3d!([x[i], xl1[i]],[y[i], yl1[i]],[zl1[i], zl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            end
        elseif directional
            for i in 1:Nx*Ny*Nz
                plot3d!([xl1[i], xl2[i]],[yl1[i], yl2[i]],[zl1[i], zl2[i]], label = nothing, arrows = true, linecolor=:black, xlims=[-0.5, Nx+0.5], ylims=[-0.5, Ny+0.5])
            end
        end
    end
    return p
end


function ScalarOrderParameter2D(sys::LebwohlLasher2D)
    Qxx, Qxy = Qtensor2D(sys)
    return 2*sqrt(Qxx^2 + Qxy^2)
end

function ScalarOrderParameter3D(sys::LebwohlLasher3D)
    Qxx, Qyy, Qzz, Qxy, Qyz, Qzx = Qtensor3D(sys)
    Q = [Qxx Qxy Qzx ; Qxy Qyy Qyz ; Qzx Qyz Qzz]
return maximum(eigvals(Q))
end


function ScalarOrderParameter3Dmod(sys::LebwohlLasher3D)
    Qxx, Qyy, Qzz, Qxy, Qyz, Qzx = Qtensor3D(sys)
    Q = [Qxx Qxy Qzx ; Qxy Qyy Qyz ; Qzx Qyz Qzz]
return eigvals(Q)
end

function Director3D(sys::LebwohlLasher3D)
    Qxx, Qyy, Qzz, Qxy, Qyz, Qzx = Qtensor3D(sys)
    Q = [Qxx Qxy Qzx ; Qxy Qyy Qyz ; Qzx Qyz Qzz]
    S,i = findmax(eigvals(Q))
    n = eigvecs(Q)[:,i]
return n
end

function CombineBitMatrices(BitMatrix_List)
    result = BitMatrix_List[1]
    for matrix in BitMatrix_List[2:end]
        result = result .|| matrix
    end
    return BitMatrix(result)
end

function FindDefects2D(sys::LebwohlLasher2D; bitsout::Bool = false)
    Eij = EnergyDensity2D(sys, "Fixed")
    Nx = sys.Nx
    Ny = sys.Ny
    ϵ = sys.ϵ
    defectpos = []
    defectposfine = []
    neighbors = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    neighborsc = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0]]
    nextneighborsc = [[2,0],[2,1],[2,2],[1,2],[0,2],[-1,2],[-2,2],[-2,1], [-2, 0],[-2, -1],[-2, -2],[-1, -2],[0, -2],[1, -2], [2, -2], [2, -1], [2, 0]]
    nextneighborscb = [[2,0],[2,1],[1,2],[0,2],[-1,2],[-2,1], [-2, 0],[-2, -1],[-1, -2],[0, -2],[1, -2], [2, -1], [2, 0]]
    ϕdomain = [i*π/4 for i in 0:8]
    ϕdomainnn = [0, atan(1/2), pi/4, atan(2), pi/2, pi/2 + atan(1/2), 3*pi/4, pi/2 + atan(2), pi, pi + atan(1/2), 5*pi/4, pi + atan(2), 3*pi/2, 3*pi/2 + atan(1/2), 7*pi/4, 3*pi/2 + atan(2), 2*pi]

    Emax = maximum(Eij)
    ## Find areas where defects exist
    for i in 3:Nx-2
        for j in 3:Ny-2
            if Eij[i,j] > Emax - ϵ
                push!(defectpos, [i, j])
            end
        end
    end
    ## Find center of the defects
    for pt1 in defectpos
        dE = []
        for pt2 in defectpos
            if in(pt2 .- pt1, neighbors)
                push!(dE, Eij[pt1[1], pt1[2]] - Eij[pt2[1], pt2[2]])
            end
        end
        if all([dE[i] > 0 for i in 1:length(dE)])
            push!(defectposfine, pt1)
        end
    end
    ## Find charge of the defects
    θ = sys.state
    defectcharge = []
    points = []
    bitdomains = []
    for dpt in defectposfine
        hood = nextneighborsc + [dpt for p in nextneighborsc]
        # Qxx, Qxy = QtensorPoints2D(sys)
        # Qxyhood = [Qxy[hood[i][1], hood[i][2]] for i in 1:length(nextneighborsc)]
        # Qxxhood = [Qxx[hood[i][1], hood[i][2]] for i in 1:length(nextneighborsc)]
        # integrandQxx = ϕdomainnn .* Qxxhood
        # # integrandQxx = Qxxhood

        # interpolated_loop_Qxx = Spline1D(ϕdomainnn, integrandQxx, k=3)
        # integrandQxy = ϕdomainnn .* Qxyhood
        # # integrandQxy = Qxyhood

        # interpolated_loop_Qxy = Spline1D(ϕdomainnn, integrandQxy, k=3)
        # # return interpolated_loop_Qxx, interpolated_loop_Qxy
        # charge = π/2 * 1/sqrt((Dierckx.integrate(interpolated_loop_Qxy, 0, 2*pi)^2 + Dierckx.integrate(interpolated_loop_Qxx, 0, 2*pi)^2))
        # println("|charge| = ",charge)
        # if charge > 0.75
        #     charge = 0.5
        #     println("problems in the backend at pt:", dpt)
        # end

        θlist = [sys.state[hood[i][1], hood[i][2]] for i in 1:length(nextneighborsc)]
        Δθ = 0
        for (i,θ1) in enumerate(θlist[1:end-1])
            θ2 = θlist[i+1]
            if θ2/θ1 < 0
                continue
            end
            Δθ += θ2 - θ1
        end
        charge = Δθ/(2*pi)
        push!(defectcharge, round(2*charge)/2)
        push!(points, dpt)
        if bitsout
            friends = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[2,0],[2,1],[1,2],[0,2],[-1,2],[-2,1], [-2,0],[-2,-1],[-1,-2],[0,-2],[1,-2],[2,-1]]
            locale = friends + [dpt for p in friends]
            bitdomain = Matrix{Bool}(undef, size(NewSystem.state)[1], size(NewSystem.state)[2])
            bitdomain[:,:] .= false
            for idxs in locale
                bitdomain[idxs[1], idxs[2]] = true
            end
            push!(bitdomains, BitMatrix(bitdomain))
        end
    end
    if bitsout
        return bitdomains, points, defectcharge
    else 
        return points, defectcharge
    end
end    

function DivergenceQ2D(sys::LebwohlLasher2D)
    Qxx, Qxy = QtensorPoints2D(sys)
    Qyy = -1 .* Qxx
    Qyx = Qxy
    Nx = sys.Nx
    Ny = sys.Ny
    DivQx = zeros(Float64, size(Qxx))
    DivQy = zeros(Float64, size(Qxx))
    for i in 2:Nx-1
        for j in 2:Ny-1
            DivQx[i,j] = (Qxx[i+1, j] - Qxx[i-1, j])/2 + (Qxy[i, j+1] - Qxy[i, j-1])/2
            DivQy[i,j] = (Qyx[i+1, j] - Qyx[i-1, j])/2 + (Qyy[i, j+1] - Qyy[i, j-1])/2
        end
    end
    return [DivQx, DivQy]
end

function GradientQ2D(sys::LebwohlLasher2D)
    Qxx, Qxy = QtensorPoints2D(sys)
    Nx = sys.Nx
    Ny = sys.Ny
    dxQxx = zeros(Float64, size(Qxx))
    dxQxy = zeros(Float64, size(Qxx))
    dyQxy = zeros(Float64, size(Qxx))
    dyQxx = zeros(Float64, size(Qxx))
    for i in 2:Nx-1
        for j in 2:Ny-1
            dxQxx[i,j] = (Qxx[i+1, j] - Qxx[i-1, j])/2
            dyQxy[i,j] = (Qxy[i, j+1] - Qxy[i, j-1])/2
            dxQxy[i,j] = (Qxy[i+1, j] - Qxy[i-1, j])/2
            dyQxx[i,j] = (Qxx[i, j+1] - Qxx[i, j-1])/2
        end
    end
    return [dxQxx, dyQxy, dxQxy, dyQxx]
end

function GetDefectTensor2D(sys::LebwohlLasher2D, pos, charge)
    neighbors = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    nextneighbors = [[2,0],[2,1],[2,2],[1,2],[0,2],[-1,2],[-2,2],[-2,1], [-2, 0],[-2, -1],[-2, -2],[-1, -2],[0, -2],[1, -2], [2, -2], [2, -1]]
    friends = vcat(neighbors, nextneighbors)
    # friends = neighbors
    points = friends + [pos for i in 1:length(friends)]
    push!(points, pos)
    if charge == 0.5
        DivQ = DivergenceQ2D(sys)
        p = []
        for idxs in points
            v = [DivQ[1][idxs[1], idxs[2]], DivQ[2][idxs[1], idxs[2]]]
            if v == [0.0, 0.0]
                continue
            end
            v = v ./ norm(v)
            push!(p,v)
        end
        p = mean(p)
        p = p ./ norm(p)
        return p
    elseif charge == -0.5
        GradQ = GradientQ2D(sys)
        dxQxx = GradQ[1]
        dyQxy = GradQ[2]
        dxQxy = GradQ[3]
        dyQxx = GradQ[4]
        T = []
        for idxs in points
            push!(T, [[[dxQxx[idxs[1], idxs[2]], dxQxy[idxs[1], idxs[2]]], [dxQxy[idxs[1], idxs[2]], -dxQxx[idxs[1], idxs[2]]]], [[dyQxx[idxs[1], idxs[2]], dyQxy[idxs[1], idxs[2]]], [dyQxy[idxs[1], idxs[2]], -dyQxx[idxs[1], idxs[2]]]]])
        end
        T = mean(T)
        T = T ./ norm(T)
        βs = 0:0.01:2*pi/3
        dotted = []
        for β in βs
            b = [cos(β), sin(β)]
            push!(dotted, sum(sum(sum([[[T[i][j][k] * (b[i]) * (b[j]) * (b[k]) for i in 1:2] for j in 1:2] for k in 1:2]))))
        end
        max = findmax(dotted)
        ψ = βs[max[2]]
        ps = [[cos(ψ), sin(ψ)],[cos(ψ + 2*pi/3), sin(ψ + 2*pi/3)],[cos(ψ + 4*pi/3), sin(ψ + 4*pi/3)]]
        return T, ps
    end
end
    
function PlusHalfDefect(pos, orientation)
    ϕ = orientation
    if pos[1] < 0
        return 0.5*(atan((pos[2] + 0.0001)/(pos[1] + 0.0001))+pi) + ϕ
    elseif pos[1] >= 0
        return 0.5*atan((pos[2] + 0.0001)/(pos[1] + 0.0001)) + ϕ
    end
end

function MinusHalfDefect(pos, orientation)
    ϕ = orientation
    if pos[1] < 0
        return -0.5*(atan((pos[2] + 0.0001)/(pos[1] + 0.0001))+pi) + ϕ
    elseif pos[1] >= 0
        return -0.5*atan((pos[2] + 0.0001)/(pos[1] + 0.0001)) + ϕ
    end
end

function InitializeDefect2D!(sys::LebwohlLasher2D, pos, charge, orientation)
    reg = vec(sys.grid.MeshGridCoords)
    points = reg - [pos for p in reg]
    if charge == 0.5
        for (id,idxs) in enumerate(points)
            sys.state[reg[id][1], reg[id][2]] = wrap_angle(sys.state[reg[id][1], reg[id][2]] + PlusHalfDefect(idxs, orientation))
        end
    elseif charge == -0.5       
        for (id,idxs) in enumerate(points)
            sys.state[reg[id][1], reg[id][2]] = wrap_angle(sys.state[reg[id][1], reg[id][2]] + MinusHalfDefect(idxs, orientation))
        end
    end

    friends = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[2,0],[2,1],[1,2],[0,2],[-1,2],[-2,1], [-2,0],[-2,-1],[-1,-2],[0,-2],[1,-2],[2,-1]]
    locale = friends + [pos for p in friends]
    fixed = Matrix{Bool}(undef, size(sys.state)[1], size(sys.state)[2])
    fixed[:,:] .= false
    for idxs in locale
        fixed[idxs[1], idxs[2]] = true
    end

    return BitMatrix(fixed), pos, charge
end

function DefectForce2D!(sys::LebwohlLasher2D, Defect::BitMatrix; BC::String = "Free", fix::BitMatrix = BitMatrix(zeros(Bool, sys.Nx, sys.Ny)), InteractionPotential = PairwiseEnergy2D)
    Nx = sys.Nx
    Ny = sys.Ny

    ## Take the entropy to zero and evolve the system into its free energy minimum
    sys.T = 0.00001
    MCStepper2D!(sys, 100, 1.0, "Periodic", fix = Defect .|| fix, InteractionPotential=InteractionPotential)

    # steps: calculate forces by measuring the change in free energy of the system after the defect is repositioned and a solution is converged for the surroundings
    points = [[i,j] for i in 1:sys.Nx, j in 1:sys.Ny if Defect[i,j]]

    # step X direction # step -X direction
    trialsystemR = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystemL = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystemR.state = copy(sys.state)
    trialsystemL.state = copy(sys.state)
    trialDefectR = BitMatrix(Defect)
    trialDefectL = BitMatrix(Defect)
   
    for i in 1:Nx
        for j in 1:Ny
            trialDefectR[i,j] = Defect[wrap_index(i+1, Nx),j]
            trialDefectL[i,j] = Defect[wrap_index(i-1, Nx),j]
        end
    end
    for idxs in points
        trialsystemR.state[wrap_index(idxs[1] + 1, Nx), idxs[2]] = sys.state[idxs[1], idxs[2]]
        trialsystemL.state[wrap_index(idxs[1] - 1, Nx), idxs[2]] = sys.state[idxs[1], idxs[2]]
    end
    
    MCStepper2D!(trialsystemR, 100, 1.0, "Periodic", fix = trialDefectR .|| fix, InteractionPotential=InteractionPotential)
    MCStepper2D!(trialsystemL, 100, 1.0, "Periodic", fix = trialDefectL .|| fix, InteractionPotential=InteractionPotential)
    ELstep = FreeEnergy2D(trialsystemL, "Periodic", InteractionPotential=InteractionPotential)
    ERstep = FreeEnergy2D(trialsystemR, "Periodic", InteractionPotential=InteractionPotential)
 
    ΔEx = (ERstep - ELstep)/2

    # step Y direction # step -Y direction
    trialsystemU = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystemD = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystemU.state = copy(sys.state)
    trialsystemD.state = copy(sys.state)
    trialDefectU = BitMatrix(Defect)
    trialDefectD = BitMatrix(Defect)
    
    for i in 1:Nx
        for j in 1:Ny
            trialDefectU[i,j] = Defect[i, wrap_index(j+1, Ny)]
            trialDefectD[i,j] = Defect[i, wrap_index(j-1, Ny)]
        end
    end
    for idxs in points
        trialsystemU.state[idxs[1], wrap_index(idxs[2]+1, Ny)] = sys.state[idxs[1], idxs[2]]
        trialsystemD.state[idxs[1], wrap_index(idxs[2]-1, Ny)] = sys.state[idxs[1], idxs[2]]
    end
    
    MCStepper2D!(trialsystemU, 100, 1.0, "Periodic", fix = trialDefectU .|| fix, InteractionPotential=InteractionPotential)
    MCStepper2D!(trialsystemD, 100, 1.0, "Periodic", fix = trialDefectD .|| fix, InteractionPotential=InteractionPotential)
    EUstep = FreeEnergy2D(trialsystemU, "Periodic", InteractionPotential=InteractionPotential)
    EDstep = FreeEnergy2D(trialsystemD, "Periodic", InteractionPotential=InteractionPotential)

    ΔEy = (EUstep - EDstep)/2
    return [-ΔEx, -ΔEy]
end

function DefectTorque(sys::LebwohlLasher2D, Defect::BitMatrix; fix::BitMatrix = BitMatrix(zeros(Bool, sys.Nx, sys.Ny)), InteractionPotential = PairwiseEnergy2D)
    # steps: calculate torques by measuring the change in free energy of the system after the defect is reoriented and a solution is converged for the surroundings
    Nx = sys.Nx
    Ny = sys.Ny
    ## Take the entropy to zero and evolve the system into its free energy minimum
    sys.T = 0.00001
    MCStepper2D!(sys, 100, 1.0, "Periodic", fix = Defect .|| fix, InteractionPotential=InteractionPotential)
    neighbors = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0]]
    friends = [[2,0],[2,2],[0,2],[-2,2],[-2,0],[-2,-2],[0,-2],[2,-2],[2,0]]
    HOA = [[2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],[2,-1]]
    points = [[i,j] for i in 1:sys.Nx, j in 1:sys.Ny if Defect[i,j]]
    pos = Int.(mean(points))

    # step X direction # step -X direction
    trialsystemC = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystemCC = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystemC.state = copy(sys.state)
    trialsystemCC.state = copy(sys.state)
    trialDefectC = BitMatrix(Defect)
    trialDefectCC = BitMatrix(Defect)

    for i in 1:length(HOA)
        trialDefectC[pos[1]+HOA[i][1],pos[2]+HOA[i][2]] = false
        trialDefectCC[pos[1]+HOA[i][1],pos[2]+HOA[i][2]] = false
    end
   
    for i in 1:length(neighbors)-1
        idxc1 = pos .+ neighbors[i]
        idxc2 = pos .+ neighbors[i+1]
        idxcc1 = pos .+ reverse(neighbors)[i]
        idxcc2 = pos .+ reverse(neighbors)[i+1]
        trialsystemC.state[idxc2[1], idxc2[2]] = sys.state[idxc1[1], idxc1[2]]
        trialsystemCC.state[idxcc2[1], idxcc2[2]] = sys.state[idxcc1[1], idxcc1[2]]
    end

    for i in 1:length(friends)-1
        idxc1 = pos .+ friends[i]
        idxc2 = pos .+ friends[i+1]
        idxcc1 = pos .+ reverse(friends)[i]
        idxcc2 = pos .+ reverse(friends)[i+1]
        trialsystemC.state[idxc2[1], idxc2[2]] = sys.state[idxc1[1], idxc1[2]]
        trialsystemCC.state[idxcc2[1], idxcc2[2]] = sys.state[idxcc1[1], idxcc1[2]]
    end
    
    MCStepper2D!(trialsystemC, 100, 1.0, "Periodic", fix = trialDefectC .|| fix, InteractionPotential=InteractionPotential)
    MCStepper2D!(trialsystemCC, 100, 1.0, "Periodic", fix = trialDefectCC .|| fix, InteractionPotential=InteractionPotential)
    ECstep = FreeEnergy2D(trialsystemC, "Periodic", InteractionPotential=InteractionPotential)
    ECCstep = FreeEnergy2D(trialsystemCC, "Periodic", InteractionPotential=InteractionPotential)
    
    ## Needs some more thought

    ΔEθ = (ECCstep - ECstep)/(π/2)
    return ΔEθ
end

function OpticalTweezer!(sys::LebwohlLasher2D, Defect::BitMatrix, Instructions::String; fix::BitMatrix = BitMatrix(zeros(Bool, sys.Nx, sys.Ny)), InteractionPotential = PairwiseEnergy2D)
    Nx = sys.Nx
    Ny = sys.Ny
    neighbors = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0]]
    friends = [[2,0],[2,2],[0,2],[-2,2],[-2,0],[-2,-2],[0,-2],[2,-2],[2,0]]
    HOA = [[2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],[2,-1]]
    points = [[i,j] for i in 1:sys.Nx, j in 1:sys.Ny if Defect[i,j]]
    pos = Int.(mean(points))
    trialDefect = BitMatrix(Defect)
    trialsystem = LebwohlLasher2D(Mesh(sys.Nx, sys.Ny), sys.T, sys.ϵ, "Uniform")
    trialsystem.state = copy(sys.state)

    if Instructions == "Rotate CC" # Perform a rotation CC by π/4
        for i in 1:length(HOA)
            trialDefect[pos[1]+HOA[i][1],pos[2]+HOA[i][2]] = false
        end
       
        for i in 1:length(neighbors)-1
            idxcc1 = pos .+ reverse(neighbors)[i]
            idxcc2 = pos .+ reverse(neighbors)[i+1]
            trialsystem.state[idxcc2[1], idxcc2[2]] = sys.state[idxcc1[1], idxcc1[2]]
        end
    
        for i in 1:length(friends)-1
            idxcc1 = pos .+ reverse(friends)[i]
            idxcc2 = pos .+ reverse(friends)[i+1]
            trialsystem.state[idxcc2[1], idxcc2[2]] = sys.state[idxcc1[1], idxcc1[2]]
        end
        MCStepper2D!(trialsystem, 100, 1.0, "Free", fix = trialDefect .|| fix, InteractionPotential=InteractionPotential)
        sys.state = copy(trialsystem.state)
        
    elseif Instructions == "Rotate CC"
    end
end

