using Combinatorics  # For combinations
using WignerSymbols  # For wigner3j symbols, https://github.com/Jutho/WignerSymbols.jl
using SparseArrays   # For sparse arrays

############################################## Bit functions ##############################################

# From Hacker's Delight, p.66: Counts # 1s in base-10 number x.
# Counts the number of 1s in state/number x (physically, the number of occupied sites in the state x). 

function countBits(x::Int)               
    x = x - ((x >> 1) & 0x55555555)    
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F
end

# Same as countBits but weighted. Physically, counts spins up - spins down. Note that half_L here means the physical length and for half filling half_L = N = 2s+1.
# Starting from the left, the first binary half = spins up and the remainder = spins down. Ex: 1011- 1st half = 10, 2nd half = 11.

function countSignedBits(x::Int, half_L::Int)
    @assert ( ndigits(x, base=2) <= 2*half_L ) "Unequal spin-up/down length"
    spin_up = x >> half_L
    spin_down = x & ((1 << half_L) - 1)
    occupied_up = countBits(spin_up)  
    occupied_down = countBits(spin_down) 
    signed_bits = occupied_up - occupied_down
    return signed_bits
end


############################################# Basis functions #############################################

# Makes all possible combinations of placing 'N' 1s in a list of length 'L' (twice the physical length)
# Combinations(1:L, N) creates a position list where the 1s are placed. Ex: L=3, N=2: [1,2] [1,3] [2,3]

function basis_number(N::Int, lmax::Int)
    L = 2*(lmax+1)^2
    @assert (0<=N<=L) "Filling outside range"
    numbers = Int[]
    for indices in combinations(1:L, N)
        num = 0                           # Then, the idea is to put 1s in the values [i,j] in the number. 
        for index in indices              # Set the bit at the specified index to be equal to 1
            num |= 1 << (L - index)       # Takes empty array 0 and puts the 1 in the position of the index 
        end
        push!(numbers, num)
    end
    return numbers
end

# Same as before but for a given magnetization M in a list of length L (twice the physical length)
# Magnetization M = \sum_{m,\sigma} m (c^\dagger_{m,\sigma} c_{m,\sigma}) 
# That's why I split spin up and spin down part, and assign a angular momentum value depending on position i
# Notation: string is like spin,spin-1,..,-spin. Ex: 1011, 1 is 3/2, 0 is 1/2, 1 is -1/2, and 1 is -3/2.
# Here mz is twice magnetization M. Ex: L=4, 1000, M = 1/2, mz = 1. Thus, we compare 2M with mz.
# This is because I want to compare integers with integers instead of dealing with floats.
######################### VERY IMPORTANT MZ IS TWICE THE VALUE OF MAGNETIZATION M #########################

function mz_max_value(N::Int, half_L::Int)
    if (mod(N,2) == 0)
        mz_max = N*(half_L-div(N,2))
    elseif (mod(N,2) == 1)
        mz_max = N*(half_L-1) - div((N-1)^2,2)
    end
    return mz_max
end
    

function basis_number_mz(N::Int, lmax::Int, mz::Int=0)
    #mz_max = mz_max_value(N,half_L)
    # The mz_max above is twice the max magnetization for a given filling N
    #@assert (mz_max >= 0) & (-mz_max <= mz <= mz_max) "Magnetization value outside range"
    basisNumberMz = Int[]
    numbers_list = basis_number(N, lmax)
    half_L = (lmax+1)^2
    
    for number in numbers_list            
        #state_spinup = number >> half_L
        #state_spindown = number & ((1 << half_L) - 1)
        
        counting = 0
        M = 0
        while (counting < N)
            for l in 0:lmax
                for m in -l:l
                    pos = l^2+l+m 
                    
                    maskdown = 2^pos
                    if ((number & maskdown) == maskdown)
                        M += m
                        counting += 1
                    end 
                    
                    maskup = 2^(pos+half_L)
                    if ((number & maskup) == maskup)
                        M += m
                        counting += 1
                    end
                end 
            end 
        end 

        if M == mz
            push!(basisNumberMz, number)   
        end
    end
    return basisNumberMz
end

# Same as above but for the spin in the z-direction.
##################### VERY IMPORTANT SZ IS TWICE THE VALUE OF PHYSICAL SPIN-Z SZ #######################
function sz_max_value(N::Int, half_L::Int)
    @assert (0 <= N <= 2*half_L) "Filling outside range"
    if (half_L >= N)
        sz_max = N
    else 
        sz_max = 2*half_L-N
    end
    return sz_max
end

function basis_number_sz(N::Int, half_L::Int, sz::Int=0)
    sz_max = sz_max_value(N,half_L)
    # The sz_max above is twice the max spin magnetization for a given filling N
    @assert (-sz_max <= sz <= sz_max) "Spin-z value outside range"
    basisNumberSz = Int[]
    numbersList = basis_number(N,2*half_L)

    for number in numbersList
        sz_number = countSignedBits(number,half_L)
        if (sz_number == sz)
            push!(basisNumberSz, number)
        end
    end
    return basisNumberSz
end

# We can combine the angular momentum conservation mz with the spin in the z-direction sz

function basis_number_mz_sz(N::Int, half_L::Int, mz::Int=0, sz::Int=0)
    sz_max = sz_max_value(N,half_L)
    mz_max = mz_max_value(N,half_L)
    @assert (-sz_max <= sz <= sz_max) "Spin-z value outside range"
    @assert (mz_max >= 0) & (-mz_max <= mz <= mz_max) "Magnetization mz outside range"
    basisNumberMzSz = Int[]

    basisNumberMz = basis_number_mz(N,half_L,mz)

    for number in basisNumberMz
        sz_number = countSignedBits(number,half_L)
        if (sz_number == sz)
            push!(basisNumberMzSz, number)
        end
    end
    return basisNumberMzSz
end

# Gives list of numbers with filling N and length L and the basisMap mapping those numbers to a basis.

function makeBasisMap(N::Int,lmax::Int)
    basisMap = Dict()
    stateID = 0
    stateList = basis_number(N, lmax)
    for state in stateList
        stateID += 1
        basisMap[state] = stateID
    end
    return (stateList,basisMap)
end

# Same as before but for a given magnetization M where Mz= 2M.
# Remember that M is defined as M = \sum_{m,\sigma} m (c^\dagger_{m,\sigma} c_{m,\sigma})

function makeBasisMapMz(N::Int,lmax::Int,mz::Int=0)
    basisMap = Dict()
    stateID = 0
    stateList = basis_number_mz(N,lmax,mz)
    for state in stateList
        stateID += 1
        basisMap[state] = stateID
    end
    return (stateList,basisMap)
end


# Refines the basis to its parity even/odd (z2) list.

function basisZ2(stateList, half_L::Int, z2::Int)
    basisMap = Dict()
    stateID = 0 
    basisZ2 = Int[]

    for state in stateList
        state_spinup = state >> half_L
        state_spindown = state & ((1 << half_L) - 1)
        state_new = state_spindown*2^half_L + state_spinup
        if state < state_new
            stateID += 1
            basisMap[state] = stateID
            push!(basisZ2, state)
        end
        if state == state_new
            phase_up = countBits(state_spinup)
            phase_down = countBits(state_spindown)
            if (mod(phase_up,2)==1) & (mod(phase_down,2)==1)
                factor = -1
            else 
                factor = 1
            end
            if factor == z2
                stateID += 1
                basisMap[state] = stateID
                push!(basisZ2, state)
            end
        end
    end
    return (basisZ2, basisMap)
end

# Refines the basis to its particle-hole symmetric/antisymmetric (z2) list

function basisPH2(stateList, half_L::Int, ph::Int)
    basisMap = Dict()
    stateID = 0 
    basisPH = Int[]
    for state in stateList
        state_spinup = state >> half_L
        state_spindown = state & ((1 << half_L) - 1)
        state_negated = state_spindown*2^half_L+ state_spinup
        state_ph = (2^(2*half_L)-1) ⊻ (state_negated)
        if state < state_ph
            stateID += 1 
            basisMap[state] = stateID
            push!(basisPH, state)
        end

        if state == state_ph
            state_negated_spinup = state_spindown
            state_negated_spindown = state_spinup
            number_up = countBits(state_negated_spinup)
            number_down = countBits(state_negated_spindown)
            phase_ph_transf = number_down
            phase_pushingCdown = (number_up + half_L)*number_down

            phase_permutations = 0 
            for i in 0:(half_L-1)
                state_negated_spinup_i = (state_negated_spinup >> i) & 1
                if state_negated_spinup_i == 1
                    phase_permutations += half_L-1 -i
                end
                state_negated_spindown_i = (state_negated_spindown >> i) & 1
                if state_negated_spindown_i == 1
                    phase_permutations +=  half_L-1 -i
                end 
            end 

            tot_phase = phase_permutations + phase_pushingCdown + phase_ph_transf

            tot_sign = (-1)^tot_phase

            if tot_sign == ph 
                stateID += 1
                basisMap[state] = stateID
                push!(basisPH, state)
            end
        end
    end
    return (basisPH, basisMap)
end      



#### Gives the representative state_new, the overall sign factor, the number of cycles nb but also importantly whethere the state was 
#### already a representative (change = 2) or needs to apply X on it (change = 1) but also if Hamiltonian acting on it makes a 
#### representative that do not lay on the symmetry sector (change = 0). This can probably be improved.
#### There are three cases here: if state < state_new (that is the state after acting with X (state_new) is bigger or not to the original state)
#### If it is then the state that we return needs to be the same as state with no factor and na=2. Change =2 its because we will have 
#### parity^change. So if change =2 then there's no parity term (since it was already a representative). 
#### The other case is if on the contrary state > state_new. Then we return state_new as it is the representative. But also a factor 
#### which depends how we commute the up with the down part (this only depends on whethere there is odd filled number on both up/down).
#### Since we needed to act with X to get the representative a factor of change =1 is also put, and nb=2 since it is a 2-cycle.
#### Finally, if state_new == state, we get nb =1 since it is a 1-cycle and change =2 (since is a representative already). Now there's 
#### something funny: we have to see whether acting with the Hamiltonian has take us outside the symmetry sector. So we check if parity 
#### of the state is the same as the parity sector we are looking into. If it is then we dont do anything. But if it isnt then put change 
#### equals to zero. Outside the function we put a condition that if change ==0 then continue since its outside the symmetry sector.
#### Why is there the factor of parity^change? Well, you see if I consider P_{\pm} = 1 \pm X acting on X then P_{pm} X = \pm P_{pm}.
#### There's a factor of \pm coming from the parity sector that we are looking at. 
#### And so if there's a state that it is not a representative but within the symmetry sector (call it \ket{b}) then we can relate to the 
#### representative via \ket{a} = factor* X \ket{b}. The factor is there cause maybe we get funny factors when relating these two states.


function stateZ2(state, half_L::Int, parity::Int)
    state_spinup = state >> half_L
    state_spindown = state & ((1 << half_L) - 1)
    state_new = state_spindown*2^half_L + state_spinup
    
    phase_up = countBits(state_spinup)
    phase_down = countBits(state_spindown)
    
    nb = 2
    change = 1
    if state == state_new 
        nb=1
        change = 2 
        if (mod(phase_up,2)==1) & (mod(phase_down,2)==1)
            if parity == 1
                change = 0 
            end
        else
            if parity == -1
                change = 0
            end
        end
    end
    
    if state < state_new
        state_new = state
        change = 2
    end 
    # I think else if here with factor = -1 and factor 1 else
    if (mod(phase_up,2)==1) & (mod(phase_down,2)==1) & (state != state_new)
        factor = -1
    else 
        factor = 1
    end
    return (state_new, factor, nb, change)
end


################################

function statePH2(state, half_L::Int, ph::Int)
    state_spinup = state >> half_L
    state_spindown= state & ((1 << half_L) - 1)
    state_negated = state_spindown*2^half_L+ state_spinup
    state_ph = (2^(2*half_L)-1) ⊻ (state_negated)
    
    #nb = 2
    #change = 1 

    if state < state_ph
        state_return = state
        change = 2
        nb = 2
        factor = 1
    elseif state == state_ph
        nb = 1
        change = 2
        factor = 1
        state_negated_spinup = state_spindown
        state_negated_spindown = state_spinup
        number_up = countBits(state_negated_spinup)
        number_down = countBits(state_negated_spindown)
        phase_ph_transf = number_down
        phase_pushingCdown = (number_up + half_L)*number_down

        phase_permutations = 0 
        for i in 0:(half_L-1)
            state_negated_spinup_i = (state_negated_spinup >> i) & 1
            if state_negated_spinup_i == 1
                phase_permutations += half_L-1 -i
            end
            state_negated_spindown_i = (state_negated_spindown >> i) & 1
            if state_negated_spindown_i == 1
                phase_permutations +=  half_L-1 -i
            end 
        end 

        tot_phase = phase_permutations + phase_pushingCdown + phase_ph_transf

        tot_sign = (-1)^tot_phase

        if tot_sign != ph 
            change = 0
        end
        state_return = state
    else
        nb = 2
        state_return = state_ph
        change = 1

        # I think is the phase from state ph
        state_ph_spinup = state_ph >> half_L
        state_ph_spindown = state_ph & ((1 << half_L) - 1)
        state_ph_negated = state_ph_spindown*2^(half_L) + state_ph_spinup
        
        
        state_ph_negated_spinup = state_ph_spindown
        state_ph_negated_spindown = state_ph_spinup
        number_up = countBits(state_ph_negated_spinup)
        number_down = countBits(state_ph_negated_spindown)
        phase_ph_transf = number_down
        phase_pushingCdown = (number_up + half_L)*number_down

        phase_permutations = 0 
        for i in 0:(half_L-1)
            state_ph_negated_spinup_i = (state_ph_negated_spinup >> i) & 1
            if state_ph_negated_spinup_i == 1
                phase_permutations += half_L-1 -i
            end
            state_ph_negated_spindown_i = (state_ph_negated_spindown >> i) & 1
            if state_ph_negated_spindown_i == 1
                phase_permutations +=  half_L-1 -i
            end 
        end 

        tot_phase = phase_permutations + phase_pushingCdown + phase_ph_transf

        factor = (-1)^tot_phase

    end
    return (state_return, factor, nb, change)
end
    


######################################### Magnetization functions #########################################

# Gives magnetization moments for vector inputstate in the basis stateList with basisMap.
# This works for any basis with any symmetries, just need to use the right basisMap function.
# In particular, we obtain the magnetization m, m^2 and m^4.

function computeMagnetization(inputstate, stateList, basisMap, half_L::Int)
    mm = 0
    m2 = 0
    m4 = 0
    for state in stateList
        value_basis = abs(inputstate[basisMap[state]])^2
        if value_basis > 0 
            bits_state = countSignedBits(state, half_L)
            mm += 0.5*(bits_state)*value_basis
            m2 += value_basis*(0.5*bits_state)^2
            m4 += value_basis*(0.5*bits_state)^4
        end 
    end
    return (mm,m2,m4)
end

function computeOverlap(inputstate, stateList, basisMap, half_L::Int)
    mm = 0
    m2 = 0
    m4 = 0
    for state in stateList
        value_basis = abs(inputstate[basisMap[state]])^2
        if value_basis > 0 
            bits_state = countBits(state)
            mm += 0.5*(bits_state)*value_basis
            m2 += value_basis*(0.5*bits_state)^2
            m4 += value_basis*(0.5*bits_state)^4
        end 
    end
    return (mm,m2,m4)
end
    

function orbMagnetizationZZ(inputstate, stateList, basisMap, half_L::Int, mi::Int, mj::Int)
    @assert ( 0 <= mi <= half_L-1 ) "M outside the [0,N-1] range"
    @assert ( 0 <= mj <= half_L-1 ) "M outside the [0,N-1] range"
    
    mm = 0
    for state in stateList
        value_basis = inputstate[basisMap[state]]^2
        if value_basis > 0
            mmi = 0 
            mmj = 0
            up_mask = 2^(half_L+mi)
            if ((state & up_mask) == up_mask)
                mmi += 1
            end
            up_mask = 2^(half_L+mj)
            if ((state & up_mask) == up_mask)
                mmj += 1
            end
            down_mask = 2^(mi)
            if ((state & down_mask) == down_mask)
                mmi -= 1
            end
            down_mask = 2^(mj)
            if ((state & down_mask) == down_mask)
                mmj -= 1
            end
            mm += mmi*mmj*value_basis
        end
    end
    return mm
end

function orbMagnetizationZ(inputstate, stateList, basisMap, half_L::Int, mi::Int)
    @assert ( 0 <= mi <= half_L-1 ) "M outside the [0,N-1] range"
    mm = 0
    for state in stateList
        value_basis = inputstate[basisMap[state]]^2
        if value_basis > 0
            mmi = 0 
            up_mask = 2^(half_L+mi)
            if ((state & up_mask) == up_mask)
                mmi += 1
            end
            down_mask = 2^(mi)
            if ((state & down_mask) == down_mask)
                mmi -= 1
            end
            mm += mmi*value_basis
        end
    end
    return mm
end

function MagnetizationZZ_L(inputstate, stateList, basisMap, half_L::Int, orbital_l::Int)
    N = half_L
    s = (N-1)/2
    sum_m1m2 = 0
    for m1 in 0:(half_L-1)
        for m2 in 0:(half_L-1)
            factor = (-1)^(m1+m2)
            vev_m1m2 = orbMagnetizationZZ(inputstate, stateList, basisMap, half_L, m1, m2)
            vev_m1m2 *= wigner3j(Float64,s,orbital_l,s,-m1+s,0,m1-s)*wigner3j(Float64,s,orbital_l,s,-m2+s,0,m2-s)
            vev_m1m2 *= factor
            sum_m1m2 += vev_m1m2
        end
    end
    sum_m1m2 *= (2*orbital_l+1)^2 *wigner3j(Float64,s,orbital_l,s,-s,0,s)^2 /(16*pi^2)
    sum_m1m2 *= N^2
    return sum_m1m2
end

function MagnetizationZ_L(inputstate, stateList, basisMap, half_L::Int, orbital_l::Int)
    N = half_L
    s = (N-1)/2
    sum_m1 = 0 
    for m1 in 0:(half_L-1)
        factor = (-1)^m1   # Here I am missing a 2s to cause I was using the square.
        vev_m1 = orbMagnetizationZ(inputstate, stateList, basisMap, half_L, m1)
        vev_m1 *= factor*wigner3j(Float64,s,orbital_l,s,-m1+s,0,m1-s)
        sum_m1 += vev_m1
    end
    sum_m1 *= N*(2*orbital_l+1)*wigner3j(Float64,s,orbital_l,s,-s,0,s)/(4*pi)
    return sum_m1
end


function double_occupancy(inputstate, stateList, basisMap, half_L::Int, m::Int)
    @assert ( 0 <= m <= half_L-1 ) "M outside the [0,N-1] range"

    number_m = 0 
    for state in stateList
        value_basis = inputstate[basisMap[state]]^2
        if value_basis > 0
            mask = 2^m+2^(half_L+m)
            if ((state & mask) == mask)
                number_m += value_basis
            end 
        end 
    end
    return number_m
end

function total_occupancy(inputstate, stateList, basisMap, half_L::Int)
    total_number = 0 

    for m in 0:(half_L-1)
        total_number += double_occupancy(inputstate, stateList, basisMap, half_L, m)
    end
    return total_number 
end

function orb_Onsite00(inputstate, stateList, basisMap, half_L::Int, mi::Int, mj::Int)
    @assert ( 0 <= mi <= half_L-1 ) "M outside the [0,N-1] range"
    @assert ( 0 <= mj <= half_L-1 ) "M outside the [0,N-1] range"
    
    mm = 0
    for state in stateList
        value_basis = inputstate[basisMap[state]]^2
        if value_basis > 0
            mmi = 0 
            mmj = 0
            up_mask = 2^(half_L+mi)
            if ((state & up_mask) == up_mask)
                mmi += 1
            end
            up_mask = 2^(half_L+mj)
            if ((state & up_mask) == up_mask)
                mmj += 1
            end
            down_mask = 2^(mi)
            if ((state & down_mask) == down_mask)
                mmi += 1
            end
            down_mask = 2^(mj)
            if ((state & down_mask) == down_mask)
                mmj += 1
            end
            mm += mmi*mmj*value_basis
        end
    end
    return mm
end

function orb_Onsite0(inputstate, stateList, basisMap, half_L::Int, mi::Int)
    @assert ( 0 <= mi <= half_L-1 ) "M outside the [0,N-1] range"
    mm = 0
    for state in stateList
        value_basis = inputstate[basisMap[state]]^2
        if value_basis > 0
            mmi = 0 
            up_mask = 2^(half_L+mi)
            if ((state & up_mask) == up_mask)
                mmi += 1
            end
            down_mask = 2^(mi)
            if ((state & down_mask) == down_mask)
                mmi += 1
            end
            mm += mmi*value_basis
        end
    end
    return mm
end 


function onsite00_L(inputstate, stateList, basisMap, half_L::Int, orbital_l::Int)
    N = half_L
    s = (N-1)/2
    sum_m1m2 = 0
    for m1 in 0:(half_L-1)
        for m2 in 0:(half_L-1)
            factor = (-1)^(m1+m2)
            vev_m1m2 = orb_Onsite00(inputstate, stateList, basisMap, half_L, m1, m2)
            vev_m1m2 *= wigner3j(Float64,s,orbital_l,s,-m1+s,0,m1-s)*wigner3j(Float64,s,orbital_l,s,-m2+s,0,m2-s)
            vev_m1m2 *= factor
            sum_m1m2 += vev_m1m2
        end
    end
    sum_m1m2 *= (2*orbital_l+1)^2 *wigner3j(Float64,s,orbital_l,s,-s,0,s)^2 /(16*pi^2)
    sum_m1m2 *= N^2
    return sum_m1m2
end

function onsite0_L(inputstate, stateList, basisMap, half_L::Int, orbital_l::Int)
    N = half_L
    s = (N-1)/2
    sum_m1 = 0 
    for m1 in 0:(half_L-1)
        factor = (-1)^m1   # I am assuming that'll be using the square of the function. Else there's a missing 2s here. I guess I should put it.
        vev_m1 = orb_Onsite0(inputstate, stateList, basisMap, half_L, m1)
        vev_m1 *= factor*wigner3j(Float64,s,orbital_l,s,-m1+s,0,m1-s)
        sum_m1 += vev_m1
    end
    sum_m1 *= N*(2*orbital_l+1)*wigner3j(Float64,s,orbital_l,s,-s,0,s)/(4*pi)
    return sum_m1
end

    

########################################## Hamiltonian structures ##########################################

# Defining structures for the Hamiltonian to pack all the information
# I can use this for both basis_number and basis_number_mz

struct HIsing
    N::Int 
    half_L::Int
    lmax::Int
    h::Float64
    v00::Float64
    v11::Float64
    J00::Float64
    J11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsing(;N=2,half_L=2,lmax=1,h=0,v00=0,v11=0,J00=0,J11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,lmax,h,v00,v11,J00,J11,stateList,basisMap)
end

# Same structure as before but including the parity.

struct HIsingZ2
    N::Int 
    half_L::Int
    parity::Int
    h::Float64
    v00::Float64
    v11::Float64
    J00::Float64
    J11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingZ2(;N=2,half_L=2,parity=1,h=0,v00=0,v11=0,J00=0,J11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,parity,h,v00,v11,J00,J11,stateList,basisMap)
end

# Same structure as before but including the PH but not Z2 Ising.

struct HIsingPH
    N::Int 
    half_L::Int
    ph::Int
    h::Float64
    v00::Float64
    v11::Float64
    J00::Float64
    J11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingPH(;N=2,half_L=2,ph=1,h=0,v00=0,v11=0,J00=0,J11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,ph,h,v00,v11,J00,J11,stateList,basisMap)
end

struct HIsingPHZ2
    N::Int 
    half_L::Int
    ph::Int
    parity::Int
    h::Float64
    v00::Float64
    v11::Float64
    J00::Float64
    J11::Float64
    stateList::Array{Int64,1}
    basisMap::Dict
    HIsingPHZ2(;N=2,half_L=2,ph=1,parity=1,h=0,v00=0,v11=0,J00=0,J11=0,stateList=Nothing
        ,basisMap=Nothing) = new(N,half_L,ph,parity,h,v00,v11,J00,J11,stateList,basisMap)
end

# I think next is creating a structure for the parity even/odd sections as well as for the particle-hole.
    

########################################## Making the Hamiltonian ##########################################

function makeH(data::HIsing)
    N = data.N
    half_L = data.half_L
    lmax = data.lmax
    h = data.h
    v00 = data.v00
    v11 = data.v11
    J00 = data.J00
    J11 = data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    for state in stateList
        idxA = basisMap[state]
        state_spinup = state >> half_L
        state_spindown = state & ((1 << half_L) - 1)
        
        if (h != 0)
            for i in 0:half_L-1
                state_spinup_i = (state_spinup >> i) & 1
                state_spindown_i = (state_spindown >> i) & 1

                if xor(state_spinup_i, state_spindown_i) == 1
                    if state_spinup_i == 0 
                        count_upto_down = state >> (i+1)
                        bitsdown = countBits(count_upto_down)
                        count_upto_up = state_spinup >> (i+1)
                        bitsup = countBits(count_upto_up)
                        factor = (-1)^(bitsdown+bitsup)
                    else
                        count_upto_up = state_spinup >> (i+1)
                        bitsup = countBits(count_upto_up)
                        count_upto_down = state >> (i+1)
                        bitsdown = countBits(count_upto_down)
                        factor = (-1)^(bitsdown + bitsup+1)
                    end
                    
                    mask = 2^i + 2^(half_L+i)
                    stateB = xor(state, mask)
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-factor*h)
                end
            end
        end
        

        for l1 in 0:lmax
            for l2 in 0:lmax 
                for l3 in 0:lmax
                    for l4 in 0:lmax
                        lm_max = min(l1+l4,l2+l3)
                        lm_min = max(abs(l1-l4), abs(l2-l3))
                        if (lm_min>lm_max) continue end
                        for lm in lm_min:lm_max
                            for m1 in (-l1):l1
                                for m2 in (-l2):l2
                                    for m3 in (-l3):l3
                                        m4 = m1+m2-m3
                                        if ((-l4<= m4 <= l4) && (abs(m1-m4)<= lm) && (abs(m2-m3)<= lm))
                                            Ul = (v00 - v11*lm*(lm+1)/((lmax+1)^2))/N # (R = sqrt(N), N = #orbitals = (lmax+1)^2, Added 1/N)
                                            Ul *= wigner3j(Float64,lm,l1,l4,0,0,0)*wigner3j(Float64,lm,l2,l3,0,0,0)*(-1)^(m1+m3)
                                            Ul *= wigner3j(Float64,lm,l1,l4,m1-m4,-m1,m4)*wigner3j(Float64,lm,l2,l3,m2-m3,-m2,m3)
                                            Ul *= (2*lm+1)*sqrt(2*l1+1)*sqrt(2*l2+1)*sqrt(2*l3+1)*sqrt(2*l4+1)/(4*pi)

                                            if (Ul != 0)
                                                # This gives the position starting from the left
                                                pos3 = l3^2+l3+m3+1 
                                                pos2 = l2^2+l2+m2+1
                                                pos4 = l4^2+l4+m4+1
                                                pos1 = l1^2+l1+m1+1

                                                annihilation_mask = 2^(pos3-1)+ 2^(pos4-1+half_L)
                                                if ((state & annihilation_mask) == annihilation_mask) 
                                                    state_new = state ⊻ annihilation_mask
                                                    creation_mask = 2^(pos1+ half_L-1) + 2^(pos2-1)          
                                                    if ((state_new & creation_mask) == 0)               
                                                        state_new2 = state_new ⊻ creation_mask           
                                                        idxB = basisMap[state_new2]   

                                                        lower_half_new = state_new & ((1 << half_L) - 1)
                                                        state_m3 = state_spindown >> pos3
                                                        state_m2 = lower_half_new >> pos2
                                                        
                                                        upper_half_new = state_new >> half_L      
                                                        state_m4 = state_spinup >> pos4               
                                                        state_m1 = upper_half_new >> pos1          

                                                        bitstot = countBits(state_m4)                 
                                                        bitstot += countBits(state_m1)
                                                        bitstot += countBits(state_m3)
                                                        bitstot += countBits(state_m2)

                                                        factor = (-1)^bitstot
                                    
                                                        push!(cols, idxA)
                                                        push!(rows, idxB)
                                                        push!(entries, 4*factor*Ul)
                                                    end
                                                end
                                            end 
                                        end 
                                    end 
                                end 
                            end 
                        end 
                    end 
                end 
            end 
        end 
    end
    return sparse(cols,rows, entries)
end

####################################################### Hamiltonian with Z2 ##################################################

### Notice that there's this problem I had with the na/nb when we apply the Hamiltonian to states with different na and nb

function makeH(data::HIsingZ2)
    N = data.N
    half_L = data.half_L
    parity = data.parity
    h = data.h
    v00 = data.v00
    v11 = data.v11
    J00 = data.J00
    J11 = data.J11
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)      # At half-filling, we know that N = 2spin+1. We compute s = 2spin = N-1. 
    spin = s/2               # And here, we compute the value of the spin.

    V0 = (4*spin+1) * v00   # Yin's paper, potential is V = \sum_l V_l (4spin-2l+1) wigner3j*wigner3j
    V1 = (4*spin-1) * v11   # Def: V0=(4spin+1)V_0, V1=(4spin-1)V_1 [including spin part on definition] 
    J0 = (4*spin+1) * J00
    J1 = (4*spin-1) * J11
                             # Setting V_1 = 1 and V_0 = 4.75 [value that Yin-Chen uses] 
    for state in stateList
        idxA = basisMap[state]
        state_spinup = state >> half_L
        state_spindown = state & ((1 << half_L) - 1)
        na = 2 
        if (state == state_spindown*2^half_L + state_spinup)
            na = 1
        end

        if (h != 0)
            for i in 0:half_L-1
                state_spinup_i = (state_spinup >> i) & 1
                state_spindown_i = (state_spindown >> i) & 1

                if xor(state_spinup_i, state_spindown_i) == 1
                    if state_spinup_i == 0 
                        count_upto_down = state >> (i+1)
                        bitsdown = countBits(count_upto_down)
                        count_upto_up = state_spinup >> (i+1)
                        bitsup = countBits(count_upto_up)
                        factor = (-1)^(bitsdown+bitsup)
                    else
                        count_upto_up = state_spinup >> (i+1)
                        bitsup = countBits(count_upto_up)
                        count_upto_down = state >> (i+1)
                        bitsdown = countBits(count_upto_down)
                        factor = (-1)^(bitsdown + bitsup+1)
                    end 

                    mask = 2^i + 2^(half_L+i)
                    stateBs = xor(state, mask)
                    #println("StateBs: ", stateBs)
                    (stateB,fact,nb,change) = stateZ2(stateBs, half_L, parity)
                    ##############
                    #println("State in: ", state, " and state out:", stateB)
                    if change == 0
                        continue
                    end
                    idxB = basisMap[stateB]
                    push!(cols,idxA)
                    push!(rows,idxB)
                    push!(entries,-factor*h*fact*sqrt(na/nb)*parity^change)
                    #println(-factor*h*fact*parity*sqrt(nb/na))
                end
            end
        end

        for m1 in 0:s
            for m2 in 0:s
                for m3 in 0:s
                    m4 = m1 + m2 - m3
                    if (0<= m4 <= s)
                        Vl = 0
                        Jl = 0
                        wignerl0 = wigner3j(Float64,spin,spin,s,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m4-spin,m3-spin,-m3-m4+s)
                        #wignerl0 += wigner3j(Float64,spin,spin,s,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m3-spin,m4-spin,-m3-m4+s)
                        wignerl1 = 0 
                        if (abs(-m1-m2+s) <= s-1) && (abs(-m3-m4+s) <= s-1)
                            wignerl1 = wigner3j(Float64,spin,spin,s-1,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m4-spin,m3-spin,-m3-m4+s)
                            #wignerl1 += wigner3j(Float64,spin,spin,s-1,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m3-spin,m4-spin,-m3-m4+s)
                        end
                        Vl += V0*wignerl0
                        Vl += V1*wignerl1
                        Jl += J0*wignerl0
                        Jl += J1*wignerl1

                        Vuu = (Vl - Jl)
                        Vud = Vl + Jl

                        if (Vud != 0)
                            annihilation_mask = 2^(m4+ half_L) + 2^m3           
                            if ((state & annihilation_mask) == annihilation_mask)  
                                state_new = state ⊻ annihilation_mask              
                                creation_mask = 2^(m1+ half_L) + 2^m2          
                                if ((state_new & creation_mask) == 0)               
                                    state_new2s = state_new ⊻ creation_mask 
                                    ######################
                                    #println(state_new2s)
                                    (state_new2,fact,nb,change) = stateZ2(state_new2s, half_L, parity)
                                    if change == 0       #### Condition that H has taken state outside of the symmetry sector
                                        continue 
                                    end
                                    #println(state_new2)
                                    idxBB = basisMap[state_new2]                    
                                    
                                    upper_half_new = state_new >> half_L      
                                    state_m4 = state_spinup >> (m4+1)               
                                    state_m1 = upper_half_new >> (m1+1)           

                                    lower_half_new = state_new & ((1 << half_L) - 1)  
                                    state_m3 = state_spindown >> (m3+1)                      
                                    state_m2 = lower_half_new >> (m2+1)                   

                                    bitstot = countBits(state_m4)                 
                                    bitstot += countBits(state_m1)
                                    bitstot += countBits(state_m3)
                                    bitstot += countBits(state_m2)

                                    factor = (-1)^bitstot
                                    
                                    push!(cols, idxA)
                                    push!(rows, idxBB)
                                    push!(entries, factor*Vud*fact*sqrt(na/nb)*parity^change)  
                                end
                            end
                        end
                        if (Vuu != 0)
                            if (m2 == m4)
                                annihilation_upmask = 2^(m3+half_L)
                                if ((annihilation_upmask & state) == annihilation_upmask)
                                    state_new = state ⊻ annihilation_upmask
                                    creation_upmask = 2^(m1+half_L)
                                    if ((creation_upmask & state_new) == 0)
                                        state_new2s = state_new ⊻ creation_upmask
                                        ###########################
                                        (state_new2,fact,nb,change) = stateZ2(state_new2s, half_L, parity)
                                        if change == 0
                                            continue 
                                        end
                                        idxBB = basisMap[state_new2]
                                        state_m3 = state_spinup>>(m3+1)
                                        state_m1 = state_new >> (m1+half_L+1)
                                        bitstot = countBits(state_m3)
                                        bitstot += countBits(state_m1)
                                        
                                        factor = (-1)^bitstot

                                        push!(cols,idxA)
                                        push!(rows,idxBB)
                                        push!(entries,fact*(parity^change)*sqrt(na/nb)*factor*Vuu/2)
                                    end 
                                end

                                annihilation_downmask = 2^(m3)
                                if ((annihilation_downmask & state) == annihilation_downmask)
                                    state_new = state ⊻ annihilation_downmask
                                    creation_downmask = 2^(m1)
                                    if ((creation_downmask & state_new) == 0)
                                        state_new2s = state_new ⊻ creation_downmask
                                        #################################
                                        (state_new2,fact,nb,change) = stateZ2(state_new2s, half_L, parity)
                                        if change == 0
                                            continue 
                                        end
                                        idxBB = basisMap[state_new2]
                                        state_m3 = state_spindown>>(m3+1)
                                        lower_half_new = state_new & ((1 << half_L) - 1)
                                        state_m1 = lower_half_new >> (m1+1)
                                        bitstot = countBits(state_m3)
                                        bitstot += countBits(state_m1)
                                        
                                        factor = (-1)^bitstot

                                        push!(cols,idxA)
                                        push!(rows,idxBB)
                                        push!(entries,fact*(parity^change)*sqrt(na/nb)*factor*Vuu/2)
                                    end 
                                end
                            end

                            if (m4 != m3)
                                annhilation_upmask =  2^(m4+half_L) + 2^(m3+half_L)
                                if ((state & annhilation_upmask) == annhilation_upmask)
                                    state_new = state ⊻ annhilation_upmask
                                    if m1 != m2
                                        creation_upmask = 2^(m1+half_L) + 2^(m2+half_L)
                                        if ((creation_upmask & state_new) == 0 )
                                            state_new2s = state_new ⊻ creation_upmask
                                            #######################
                                            (state_new2,fact,nb,change) = stateZ2(state_new2s, half_L, parity)
                                            if change == 0 
                                                continue 
                                            end
                                            idxBB = basisMap[state_new2]
                                            state_new1 = state ⊻ 2^(m4+half_L)
                                            state_new02 = state_new ⊻ 2^(m2+half_L)
                                    
                                            upper_half_new = state_new >> half_L        # This is for m2
                                            upper_half_new1 = state_new1 >> half_L      # This is for m3
                                            upper_half_new2 = state_new02 >> half_L     # This is for m1

                                            state_m4 = state_spinup >> (m4+1)                
                                            state_m3 = upper_half_new1 >> (m3+1)             
                                            state_m2 = upper_half_new >> (m2+1)
                                            state_m1 = upper_half_new2 >> (m1+1)
                                            
                                            bitstot = countBits(state_m4)                  
                                            bitstot += countBits(state_m1)
                                            bitstot += countBits(state_m3)
                                            bitstot += countBits(state_m2)
                                            factor = (-1)^bitstot

                                            push!(cols,idxA)
                                            push!(rows,idxBB)
                                            push!(entries, fact*(parity^change)*sqrt(na/nb)*factor*Vuu/2)
                                        end
                                    end 
                                end      
                                annhilation_downmask = 2^(m4) + 2^(m3)
                                if ((state & annhilation_downmask) == annhilation_downmask)
                                    state_new = state ⊻ annhilation_downmask
                                    if m1 != m2
                                        creation_downmask = 2^(m1) + 2^(m2)
                                        if ((creation_downmask & state_new) == 0 )
                                            state_new2s = state_new ⊻ creation_downmask
                                            ##################################
                                            (state_new2,fact,nb,change) = stateZ2(state_new2s, half_L, parity)
                                            if change == 0 
                                                continue
                                            end 
                                            idxBB = basisMap[state_new2]
                                            
                                            state_new1 = state ⊻ 2^(m4)
                                            state_new02 = state_new ⊻ 2^(m2)
                                            
                                            lower_half_new = state_new & ((1 << half_L) - 1)  
                                            lower_half_new1 = state_new1 & ((1 << half_L) - 1)
                                            lower_half_new2 = state_new02 & ((1 << half_L) - 1)
                                            
                                            state_m4 = state_spindown >> (m4+1)               
                                            state_m3 = lower_half_new1 >> (m3+1)
                                            state_m2 = lower_half_new >> (m2+1)                  
                                            state_m1 = lower_half_new2 >> (m1+1)

                                            bitstot = countBits(state_m4)              
                                            bitstot += countBits(state_m1)
                                            bitstot += countBits(state_m3)
                                            bitstot += countBits(state_m2)
                                            factor = (-1)^bitstot
                                            
                                            push!(cols,idxA)
                                            push!(rows,idxBB)
                                            push!(entries, fact*(parity^change)*sqrt(na/nb)*factor*Vuu/2)
                                        end
                                    end 
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end


####################################################### Hamiltonian with PH ##################################################

function makeH(data::HIsingPH)
    N = data.N
    half_L = data.half_L
    ph = data.ph
    h = data.h
    v00 = data.v00
    epsilon = data.epsilon
    stateList = data.stateList
    basisMap = data.basisMap

    cols = Vector{Int64}(undef,0)
    rows = Vector{Int64}(undef,0)
    entries = Vector{Float64}(undef, 0)

    s = (half_L-1)      # At half-filling, we know that N = 2spin+1. We compute s = 2spin = N-1. 
    spin = s/2               # And here, we compute the value of the spin.

    V0 = (4*spin+1) * v00   # Yin's paper, potential is V = \sum_l V_l (4spin-2l+1) wigner3j*wigner3j
    V1 = (4*spin-1)          # Def: V0=(4spin+1)V_0, V1=(4spin-1)V_1 [including spin part on definition] 
                             # Setting V_1 = 1 and V_0 = 4.75 [value that Yin-Chen uses] 
    for state in stateList
        idxA = basisMap[state]
        state_spinup = state >> half_L
        state_spindown = state & ((1 << half_L) - 1)
        na = 2 
        if (state == ((2^(2*half_L)-1) ⊻ (state_spindown*(2^half_L)+ state_spinup)) )
            na = 1
        end

        for i in 0:half_L-1
            state_spinup_i = (state_spinup >> i) & 1
            state_spindown_i = (state_spindown >> i) & 1

            if xor(state_spinup_i, state_spindown_i) == 1
                if state_spinup_i == 0 
                    count_upto_down = state >> (i+1)
                    bitsdown = countBits(count_upto_down)
                    count_upto_up = state_spinup >> (i+1)
                    bitsup = countBits(count_upto_up)
                    factor = (-1)^(bitsdown+bitsup)
                else
                    count_upto_up = state_spinup >> (i+1)
                    bitsup = countBits(count_upto_up)
                    count_upto_down = state >> (i+1)
                    bitsdown = countBits(count_upto_down)
                    factor = (-1)^(bitsdown + bitsup+1)
                end 

                mask = 2^i + 2^(half_L+i)
                stateBs = xor(state, mask)
                #println("StateBs: ", stateBs)
                (stateB,fact,nb,change) = statePH2(stateBs, half_L, ph)
                ##############
                #println("State in: ", state, " and state out:", stateB)
                if change == 0
                    continue
                end
                idxB = basisMap[stateB]
                push!(cols,idxA)
                push!(rows,idxB)
                push!(entries,-factor*h*fact*sqrt(na/nb)*ph^change)
                #println(-factor*h*fact*parity*sqrt(nb/na))
            end
        end

        for m1 in 0:s
            for m2 in 0:s
                for m3 in 0:s
                    m4 = m1 + m2 - m3
                    if (0<= m4 <= s)
                        Vl = 0
                        wignerl0 = wigner3j(Float64,spin,spin,s,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m4-spin,m3-spin,-m3-m4+s)
                        wignerl0 += wigner3j(Float64,spin,spin,s,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s,m3-spin,m4-spin,-m3-m4+s)
                        wignerl1 = 0 
                        if (abs(-m1-m2+s) <= s-1) && (abs(-m3-m4+s) <= s-1)
                            wignerl1 = wigner3j(Float64,spin,spin,s-1,m1-spin,m2-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m4-spin,m3-spin,-m3-m4+s)
                            wignerl1 += wigner3j(Float64,spin,spin,s-1,m2-spin,m1-spin,-m1-m2+s)*wigner3j(Float64,spin,spin,s-1,m3-spin,m4-spin,-m3-m4+s)
                        end
                        Vl += V0*wignerl0
                        Vl += V1*wignerl1

                        if (Vl != 0)
                            annihilation_mask = 2^(m4+ half_L) + 2^m3           
                            if ((state & annihilation_mask) == annihilation_mask)  
                                state_new = state ⊻ annihilation_mask              
                                creation_mask = 2^(m1+ half_L) + 2^m2          
                                if ((state_new & creation_mask) == 0)               
                                    state_new2s = state_new ⊻ creation_mask 
                                    ######################
                                    #println(state_new2s)
                                    (state_new2,fact,nb,change) = statePH2(state_new2s, half_L, ph)
                                    if change == 0       #### Condition that H has taken state outside of the symmetry sector
                                        continue 
                                    end
                                    #println(state_new2)
                                    idxBB = basisMap[state_new2]                    
                                    
                                    upper_half_new = state_new >> half_L      
                                    state_m4 = state_spinup >> (m4+1)               
                                    state_m1 = upper_half_new >> (m1+1)           

                                    lower_half_new = state_new & ((1 << half_L) - 1)  
                                    state_m3 = state_spindown >> (m3+1)                      
                                    state_m2 = lower_half_new >> (m2+1)                   

                                    bitstot = countBits(state_m4)                 
                                    bitstot += countBits(state_m1)
                                    bitstot += countBits(state_m3)
                                    bitstot += countBits(state_m2)

                                    factor = (-1)^bitstot
                                    
                                    push!(cols, idxA)
                                    push!(rows, idxBB)
                                    push!(entries, factor*Vl*(1+epsilon)*fact*sqrt(na/nb)*ph^change)  
                                end
                            end

                            if (m2 == m4) & (epsilon != 0)
                                annihilation_upmask = 2^(m3+half_L)
                                if ((annihilation_upmask & state) == annihilation_upmask)
                                    state_new = state ⊻ annihilation_upmask
                                    creation_upmask = 2^(m1+half_L)
                                    if ((creation_upmask & state_new) == 0)
                                        state_new2s = state_new ⊻ creation_upmask
                                        ###########################
                                        (state_new2,fact,nb,change) = statePH2(state_new2s, half_L, ph)
                                        if change == 0
                                            continue 
                                        end
                                        idxBB = basisMap[state_new2]
                                        state_m3 = state_spinup>>(m3+1)
                                        state_m1 = state_new >> (m1+half_L+1)
                                        bitstot = countBits(state_m3)
                                        bitstot += countBits(state_m1)
                                        
                                        factor = (-1)^bitstot

                                        push!(cols,idxA)
                                        push!(rows,idxBB)
                                        push!(entries,fact*(ph^change)*sqrt(na/nb)*factor*epsilon*Vl/2)
                                    end 
                                end

                                annihilation_downmask = 2^(m3)
                                if ((annihilation_downmask & state) == annihilation_downmask)
                                    state_new = state ⊻ annihilation_downmask
                                    creation_downmask = 2^(m1)
                                    if ((creation_downmask & state_new) == 0)
                                        state_new2s = state_new ⊻ creation_downmask
                                        #################################
                                        (state_new2,fact,nb,change) = statePH2(state_new2s, half_L, ph)
                                        if change == 0
                                            continue 
                                        end
                                        idxBB = basisMap[state_new2]
                                        state_m3 = state_spindown>>(m3+1)
                                        lower_half_new = state_new & ((1 << half_L) - 1)
                                        state_m1 = lower_half_new >> (m1+1)
                                        bitstot = countBits(state_m3)
                                        bitstot += countBits(state_m1)
                                        
                                        factor = (-1)^bitstot

                                        push!(cols,idxA)
                                        push!(rows,idxBB)
                                        push!(entries,fact*(ph^change)*sqrt(na/nb)*factor*epsilon*Vl/2)
                                    end 
                                end
                            end

                            if (m4 != m3) & (epsilon != 0)
                                annhilation_upmask =  2^(m4+half_L) + 2^(m3+half_L)
                                if ((state & annhilation_upmask) == annhilation_upmask)
                                    state_new = state ⊻ annhilation_upmask
                                    if m1 != m2
                                        creation_upmask = 2^(m1+half_L) + 2^(m2+half_L)
                                        if ((creation_upmask & state_new) == 0 )
                                            state_new2s = state_new ⊻ creation_upmask
                                            #######################
                                            (state_new2,fact,nb,change) = statePH2(state_new2s, half_L, ph)
                                            if change == 0 
                                                continue 
                                            end
                                            idxBB = basisMap[state_new2]
                                            state_new1 = state ⊻ 2^(m4+half_L)
                                            state_new02 = state_new ⊻ 2^(m2+half_L)
                                    
                                            upper_half_new = state_new >> half_L        # This is for m2
                                            upper_half_new1 = state_new1 >> half_L      # This is for m3
                                            upper_half_new2 = state_new02 >> half_L     # This is for m1

                                            state_m4 = state_spinup >> (m4+1)                
                                            state_m3 = upper_half_new1 >> (m3+1)             
                                            state_m2 = upper_half_new >> (m2+1)
                                            state_m1 = upper_half_new2 >> (m1+1)
                                            
                                            bitstot = countBits(state_m4)                  
                                            bitstot += countBits(state_m1)
                                            bitstot += countBits(state_m3)
                                            bitstot += countBits(state_m2)
                                            factor = (-1)^bitstot

                                            push!(cols,idxA)
                                            push!(rows,idxBB)
                                            push!(entries, fact*(ph^change)*sqrt(na/nb)*factor*epsilon*Vl/2)
                                        end
                                    end 
                                end      
                                annhilation_downmask = 2^(m4) + 2^(m3)
                                if ((state & annhilation_downmask) == annhilation_downmask)
                                    state_new = state ⊻ annhilation_downmask
                                    if m1 != m2
                                        creation_downmask = 2^(m1) + 2^(m2)
                                        if ((creation_downmask & state_new) == 0 )
                                            state_new2s = state_new ⊻ creation_downmask
                                            ##################################
                                            (state_new2,fact,nb,change) = statePH2(state_new2s, half_L, ph)
                                            if change == 0 
                                                continue
                                            end 
                                            idxBB = basisMap[state_new2]
                                            
                                            state_new1 = state ⊻ 2^(m4)
                                            state_new02 = state_new ⊻ 2^(m2)
                                            
                                            lower_half_new = state_new & ((1 << half_L) - 1)  
                                            lower_half_new1 = state_new1 & ((1 << half_L) - 1)
                                            lower_half_new2 = state_new02 & ((1 << half_L) - 1)
                                            
                                            state_m4 = state_spindown >> (m4+1)               
                                            state_m3 = lower_half_new1 >> (m3+1)
                                            state_m2 = lower_half_new >> (m2+1)                  
                                            state_m1 = lower_half_new2 >> (m1+1)

                                            bitstot = countBits(state_m4)              
                                            bitstot += countBits(state_m1)
                                            bitstot += countBits(state_m3)
                                            bitstot += countBits(state_m2)
                                            factor = (-1)^bitstot
                                            
                                            push!(cols,idxA)
                                            push!(rows,idxBB)
                                            push!(entries, fact*(ph^change)*sqrt(na/nb)*factor*epsilon*Vl/2)
                                        end
                                    end 
                                end
                            end
                        end
                    end
                end
            end
        end                                              
        
    end
    return sparse(cols,rows, entries)
end
