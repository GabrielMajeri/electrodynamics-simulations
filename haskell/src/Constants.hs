module Constants
  ( c,
    omega,
    numParticles,
    angularVelocity,
    lambda,
    radialIndex,
    azimuthalIndex,
    waistRadius,
    diskRadius,
    tau0,
    phi0,
    integrationStartTime,
    integrationEndTime,
    integrationTimeStep,
    numIntegrationSteps,
    chargeToMassRatio,
    wavelength,
    amplitude,
    polarizationX,
    polarizationY,
  )
where

import Data.Complex (Complex ((:+)))
import Types (ComplexT, RealT)

numParticles :: Integer
numParticles = 1 * 1024

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

azimuthalIndex :: Integer
azimuthalIndex = -2

waistRadius :: RealT
waistRadius = 75 * lambda

diskRadius :: RealT
diskRadius = (1.75 + (fromIntegral radialIndex)) * waistRadius

tau0 :: RealT
tau0 = 10 / omega

phi0 :: RealT
phi0 = 3 * tau0

integrationStartTime :: RealT
integrationStartTime = 0

integrationEndTime :: RealT
integrationEndTime = 6 * tau0

integrationDuration :: RealT
integrationDuration = integrationEndTime - integrationStartTime

integrationTimeStep :: RealT
integrationTimeStep = 1e-1

numIntegrationSteps :: Int
numIntegrationSteps = ceiling (integrationDuration / integrationTimeStep)

particleCharge :: RealT
particleCharge = -1

particleMass :: RealT
particleMass = 1

chargeToMassRatio :: RealT
chargeToMassRatio = particleCharge / particleMass

a0 :: RealT
a0 = 1e-2

amplitude :: RealT
amplitude = a0 * particleMass * c * omega / (abs particleCharge)

polarizationX :: ComplexT
polarizationX = (1 / (sqrt 2)) :+ 0

polarizationY :: ComplexT
polarizationY = 0 :+ (-1 / (sqrt 2))
