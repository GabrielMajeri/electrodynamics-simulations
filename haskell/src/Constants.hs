module Constants
  ( c,
    omega,
    numParticles,
    angularVelocity,
    lambda,
    radialIndex,
    waistRadius,
    diskRadius,
  )
where

import Types (RealT)

numParticles :: Integer
numParticles = 8 * 1024

c :: RealT
c = 137.036

omega :: RealT
omega = 0.057

angularVelocity :: RealT
angularVelocity = omega

lambda :: RealT
lambda = 2 * pi * c / omega

wavelength :: RealT
wavelength = lambda

radialIndex :: Integer
radialIndex = 2

waistRadius :: RealT
waistRadius = 75 * lambda

diskRadius :: RealT
diskRadius = (1.75 + (fromIntegral radialIndex)) * waistRadius
