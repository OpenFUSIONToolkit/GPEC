"""
    Special Functions

Elliptic functions and other special mathematical functions used in PENTRC.
Provides elliptic integrals of the first and second kind.
"""

"""
    ellipk(m::Float64)::Float64

Complete elliptic integral of the first kind K(m).
Computes K(m) = ∫₀^(π/2) dθ / √(1 - m·sin²θ)

# Arguments
- `m::Float64`: Modulus parameter (0 ≤ m < 1). Note: m = k² where k is the elliptic modulus.

# Returns
- `Float64`: K(m)

# Implementation
Uses polynomial approximation from NASA Technical Memorandum "Numerical Values
of the Complete Elliptic Integrals" (ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19680022044.pdf)

Accuracy: ~2×10⁻⁸ over the range [0,1)

Reference: [Logan, GPEC]
"""
function ellipk(m::Float64)::Float64
    
    # Clamp to valid range [0, 1)
    m = max(0.0, min(0.9999999, m))
    
    # Polynomial coefficients from NASA reference
    a = [1.38629436112, 0.09666344259, 0.03590092383, 0.03742563713, 0.01451196212]
    b = [0.50000000000, 0.12498593597, 0.06880248576, 0.03328355346, 0.00441787012]
    
    # Compute 1 - m
    m1 = 1.0 - m
    
    # Evaluate polynomial: K(m) = (a₀ + a₁m₁ + a₂m₁² + a₃m₁³ + a₄m₁⁴) 
    #                            + (b₀ + b₁m₁ + b₂m₁² + b₃m₁³ + b₄m₁⁴) * ln(1/m₁)
    
    m1_pow = [1.0, m1, m1^2, m1^3, m1^4]
    
    poly_a = sum(a .* m1_pow)
    poly_b = sum(b .* m1_pow)
    
    return poly_a + poly_b * log(1.0 / m1)
end


"""
    ellipe(m::Float64)::Float64

Complete elliptic integral of the second kind E(m).
Computes E(m) = ∫₀^(π/2) √(1 - m·sin²θ) dθ

# Arguments
- `m::Float64`: Modulus parameter (0 ≤ m < 1)

# Returns
- `Float64`: E(m)

# Implementation
Uses polynomial approximation from NASA Technical Memorandum.
Accuracy: ~2×10⁻⁸ over the range [0,1)

Reference: [Logan, GPEC]
"""
function ellipe(m::Float64)::Float64
    
    # Clamp to valid range [0, 1)
    m = max(0.0, min(0.9999999, m))
    
    # Polynomial coefficients from NASA reference
    a = [0.44325141463, 0.06260601220, 0.04757383546, 0.01736506451]
    b = [0.24998368310, 0.09200180037, 0.04069697526, 0.00526449639]
    
    # Compute 1 - m
    m1 = 1.0 - m
    
    # Evaluate polynomial: E(m) = (1 + a₁m₁ + a₂m₁² + a₃m₁³ + a₄m₁⁴)
    #                            + (b₁m₁ + b₂m₁² + b₃m₁³ + b₄m₁⁴) * ln(1/m₁)
    
    m1_pow = [1.0, m1, m1^2, m1^3, m1^4]
    
    poly_a = 1.0 + sum(a .* m1_pow[2:5])
    poly_b = sum(b .* m1_pow[2:5])
    
    return poly_a + poly_b * log(1.0 / m1)
end


"""
    double_factorial(n::Int)::Float64

Compute double factorial n!!
- For even n: n!! = n(n-2)(n-4)···2
- For odd n:  n!! = n(n-2)(n-4)···1

# Arguments
- `n::Int`: Non-negative integer

# Returns
- `Float64`: n!!
"""
function double_factorial(n::Int)::Float64
    if n <= 0
        return 1.0
    elseif n == 1
        return 1.0
    else
        result = 1.0
        for i in range(n, step=-2, stop=1)
            result *= i
        end
        return result
    end
end


"""
    set_fymnl()

Initialize Fourier mode coupling data.
Stub for compatibility with legacy code.
"""
function set_fymnl()
    # TODO: Load fnml data structure if needed
    return nothing
end


"""
    set_ellip()

Initialize elliptic function tables.
Stub for compatibility - Julia computes on demand.
"""
function set_ellip()
    # No-op: Julia computes elliptic functions on demand
    return nothing
end
