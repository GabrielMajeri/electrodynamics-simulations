module Lib
  ( generateInitialConditions,
    integrateTrajectories,
    computeAngularMomentaInZDirection,
  )
where

import Constants
  ( amplitude,
    angularVelocity,
    azimuthalIndex,
    c,
    chargeToMassRatio,
    diskRadius,
    integrationTimeStep,
    numIntegrationSteps,
    numParticles,
    phi0,
    radialIndex,
    tau0,
    waistRadius,
    wavelength, polarizationY, polarizationX,
  )
import Control.Monad.Random (MonadRandom (getRandomR), Rand, RandomGen)
import Data.Complex (Complex ((:+)), realPart)
import Linear (V3 (V3), V4 (V4))
import Types (Momentum (Momentum), Position (Position), RealT, Vec3, Vec4, AngularMomentum (AngularMomentum))

generateRandomPosition :: (RandomGen g) => Rand g Position
generateRandomPosition = do
  let diskRadiusSquared = diskRadius ^ 2

  -- Rejection sampling: generate u in the range [0, R^2]
  -- then compute sqrt(u)
  rSquared <- getRandomR (0, diskRadiusSquared)
  let r = sqrt rSquared

  -- Generate a random angle between 0 and 2*PI
  theta <- getRandomR (0, 2 * pi)

  -- Convert to Cartesian coordinates
  let x = r * cos (theta)
      y = r * sin (theta)

  return $ Position $ V4 0 x y 0

initialMomentum :: Momentum
initialMomentum = Momentum $ V4 1 0 0 0

generateInitialPositions :: (RandomGen g) => Int -> Rand g [Position]
generateInitialPositions n = sequence $ replicate n generateRandomPosition

generateInitialConditions :: (RandomGen g) => Rand g ([Position], [Momentum])
generateInitialConditions = do
  let n = fromIntegral numParticles
  initialPositions <- generateInitialPositions n
  let initialMomenta = replicate n initialMomentum
  return $ (initialPositions, initialMomenta)

integrateTrajectories :: ([Position], [Momentum]) -> ([Position], [Momentum])
integrateTrajectories (initialPositions, initialMomenta) =
  let variables = zip initialPositions initialMomenta
      finalVariables = map (\vars -> iterate integrationStep vars !! numIntegrationSteps) variables
   in unzip finalVariables

integrationStep :: (Position, Momentum) -> (Position, Momentum)
integrationStep (position, momentum) =
  let (Position positionVector) = position
      (V4 laboratoryTime x y z) = positionVector
      (e, b) = computeLaguerreGaussElectricAndMagneticField (V3 x y z) laboratoryTime
      coeff = cutOff (laboratoryTime - z / c)
      eMod = fmap (* coeff) e
      bMod = fmap (* coeff) b
      (Momentum momentumVector) = momentum
      acceleration = computeAcceleration momentumVector eMod bMod
      newMomentumVector = momentumVector + (fmap (* integrationTimeStep) acceleration)
      newPositionVector = positionVector + (fmap (* integrationTimeStep) newMomentumVector)
   in (Position newPositionVector, Momentum newMomentumVector)

cutOff :: RealT -> RealT
cutOff phi = exp (-t ^ 2)
  where
    t = (phi - phi0) / tau0

tolerance :: RealT
tolerance = 1e-8

computeLaguerreGaussElectricAndMagneticField :: Vec3 -> RealT -> (Vec3, Vec3)
computeLaguerreGaussElectricAndMagneticField (V3 x y z) time =
  let r = sqrt (x ^ 2 + y ^ 2)
      phi = atan2 y x
      -- z_R
      rayleighLength = pi * (waistRadius ^ 2) / wavelength
      -- w(z)
      width = waistRadius * sqrt (1 + (z / rayleighLength) ^ 2)
      -- r / w(z)
      rOverWidth = r / width
      rOverWidthSquared = rOverWidth ^ 2
      -- k
      wavenumber = 2 * pi / wavelength
      -- \|l|
      absL = abs azimuthalIndex
      -- R(z)
      radiusOfCurvature = if abs z < tolerance then 0 else z * (1 + (rayleighLength / z) ^ 2)
      -- r^2/(2 * R(z))
      curvature = if abs radiusOfCurvature < tolerance then 0 else r ^ 2 / (2 * radiusOfCurvature)
      -- \psi(z)
      gouyPhase = atan2 z rayleighLength

      sqrt2 = sqrt 2
      polynomialPart = laguerrePolynomial radialIndex (fromIntegral absL) (2 * rOverWidthSquared)
      exponentialDecay = exp (- rOverWidthSquared)
      magnitude = amplitude * (waistRadius / width) * (sqrt2 * rOverWidth) ^ absL * polynomialPart * exponentialDecay

      timePhaseShift = angularVelocity * time
      longitudinalPhaseShift = wavenumber * z
      curvaturePhaseShift = wavenumber * curvature
      azimuthalPhaseShift = (fromIntegral azimuthalIndex) * phi
      gouyPhaseShift = (2 * (fromIntegral radialIndex) + (fromIntegral absL) + 1) * gouyPhase
      totalPhaseShift = timePhaseShift - longitudinalPhaseShift + curvaturePhaseShift + azimuthalPhaseShift - gouyPhaseShift
      phase = exp (0 :+ totalPhaseShift)

      cx = x :+ 0
      cy = y :+ 0

      coeff = (magnitude :+ 0) * phase
      ex = coeff * polarizationX
      ey = coeff * polarizationY
      ez = (0 :+ 2) / ((wavenumber * width^2) :+ 0) * (cx * ex + cy * ey)

      e = V3 (realPart ex) (realPart ey) (realPart ez)

      bx = - ey / (c :+ 0)
      by = ex / (c :+ 0)
      bz = (0 :+ 1) / ((angularVelocity * width^2) :+ 0) * (cy * ex - cx * ey)

      b = V3 (realPart bx) (realPart by) (realPart bz)

   in (e, b)

laguerrePolynomial :: Integer -> RealT -> RealT -> RealT
laguerrePolynomial n _ _ | n < 0 = error "laguerrePolynomial: index cannot be negative"
laguerrePolynomial 0 _ _ = 1
laguerrePolynomial 1 alpha x = 1 + alpha - x
laguerrePolynomial 2 alpha x = 0.5 * (x^2 - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2))
laguerrePolynomial n alpha x =
  let nr = fromIntegral n
      left = laguerrePolynomial (n - 1) alpha x
      right = laguerrePolynomial (n - 2) alpha x
  in ((2 * nr - 1 + alpha - x) * left - (nr - 1 + alpha) * right) / nr

computeAcceleration :: Vec4 -> Vec3 -> Vec3 -> Vec4
computeAcceleration momentum e b =
  let V4 gamma px py pz = momentum
      V3 ex ey ez = e
      V3 bx by bz = b
      agamma = px * ex / c + py * ey / c + pz * ez / c
      ax = gamma * ex / c + py * bz - pz * by
      ay = gamma * ey / c - px * bz + pz * bx
      az = gamma * ez / c + px * by - py * bx
   in (* chargeToMassRatio) <$> (V4 agamma ax ay az)

computeAngularMomentumInZDirection :: Position -> Momentum -> AngularMomentum
computeAngularMomentumInZDirection (Position (V4 _ x y _)) (Momentum (V4 _ vx vy _)) =
  AngularMomentum $ x * vy - y * vx

computeAngularMomentaInZDirection :: [Position] -> [Momentum] -> [AngularMomentum]
computeAngularMomentaInZDirection positions momenta =
  map (uncurry computeAngularMomentumInZDirection) (zip positions momenta)
