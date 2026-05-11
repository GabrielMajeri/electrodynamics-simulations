module Lib
  ( integrateTrajectories,
    computeAngularMomentaInZDirection,
  )
where

import Constants
  ( amplitude,
    angularVelocity,
    azimuthalIndex,
    c,
    chargeToMassRatio,
    integrationTimeStep,
    numIntegrationSteps,
    phi0,
    polarizationX,
    polarizationY,
    radialIndex,
    tau0,
    waistRadius,
    wavelength,
  )
import Data.Complex (Complex ((:+)), realPart)
import Data.Vector.Unboxed (Vector)
import Data.Vector.Unboxed qualified as VU
import Linear (V3 (V3), V4 (V4))
import Types (AngularMomentum (AngularMomentum), Momentum (Momentum), Position (Position), RealT, Vec3, Vec4)

integrateTrajectories :: (Vector Position, Vector Momentum) -> (Vector Position, Vector Momentum)
integrateTrajectories (initialPositions, initialMomenta) =
  let variables = VU.zip initialPositions initialMomenta
      finalVariables = VU.map (\state -> iterate integrationStep state !! numIntegrationSteps) variables
   in VU.unzip finalVariables

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
cutOff phi = exp (-t ^ (2 :: Int))
  where
    t = (phi - phi0) / tau0

tolerance :: RealT
tolerance = 1e-8

computeLaguerreGaussElectricAndMagneticField :: Vec3 -> RealT -> (Vec3, Vec3)
computeLaguerreGaussElectricAndMagneticField (V3 x y z) time =
  let r = sqrt (x ^ (2 :: Int) + y ^ (2 :: Int))
      phi = atan2 y x
      -- z_R
      rayleighLength = pi * (waistRadius ^ (2 :: Int)) / wavelength
      -- w(z)
      width = waistRadius * sqrt (1 + (z / rayleighLength) ^ (2 :: Int))
      -- r / w(z)
      rOverWidth = r / width
      rOverWidthSquared = rOverWidth ^ (2 :: Int)
      -- k
      wavenumber = 2 * pi / wavelength
      -- \|l|
      absL = abs azimuthalIndex
      -- R(z)
      radiusOfCurvature = if abs z < tolerance then 0 else z * (1 + (rayleighLength / z) ^ (2 :: Int))
      -- r^2/(2 * R(z))
      curvature = if abs radiusOfCurvature < tolerance then 0 else r ^ (2 :: Int) / (2 * radiusOfCurvature)
      -- \psi(z)
      gouyPhase = atan2 z rayleighLength

      sqrt2 = sqrt 2
      polynomialPart = laguerrePolynomial radialIndex (fromIntegral absL) (2 * rOverWidthSquared)
      exponentialDecay = exp (-rOverWidthSquared)
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
      ez = (0 :+ 2) / ((wavenumber * width ^ (2 :: Int)) :+ 0) * (cx * ex + cy * ey)

      e = V3 (realPart ex) (realPart ey) (realPart ez)

      bx = -ey / (c :+ 0)
      by = ex / (c :+ 0)
      bz = (0 :+ 1) / ((angularVelocity * width ^ (2 :: Int)) :+ 0) * (cy * ex - cx * ey)

      b = V3 (realPart bx) (realPart by) (realPart bz)
   in (e, b)

laguerrePolynomial :: Integer -> RealT -> RealT -> RealT
laguerrePolynomial n _ _ | n < 0 = error "laguerrePolynomial: index cannot be negative"
laguerrePolynomial 0 _ _ = 1
laguerrePolynomial 1 alpha x = 1 + alpha - x
laguerrePolynomial 2 alpha x = 0.5 * (x ^ (2 :: Int) - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2))
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

computeAngularMomentaInZDirection :: (Vector Position, Vector Momentum) -> Vector AngularMomentum
computeAngularMomentaInZDirection (positions, momenta) =
  VU.zipWith computeAngularMomentumInZDirection positions momenta
