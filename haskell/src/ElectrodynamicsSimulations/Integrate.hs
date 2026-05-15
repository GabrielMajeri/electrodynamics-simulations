module ElectrodynamicsSimulations.Integrate
  ( integrateTrajectories,
  )
where

import Control.Parallel.Strategies (rdeepseq, parList, using)
import Data.Complex (Complex ((:+)))
import Data.Vector.Unboxed (Vector)
import Data.Vector.Unboxed qualified as VU
import ElectrodynamicsSimulations.Constants
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
import ElectrodynamicsSimulations.Types (Momentum (Momentum), Position (Position), RealT, Vec3, Vec4)
import GHC.Conc (numCapabilities)
import Linear (V3 (V3), V4 (V4))

data FieldConstants = FieldConstants
  { fcWavenumber :: !RealT,
    fcWaistRadiusSquared :: !RealT,
    fcRayleighLength :: !RealT,
    fcAbsL :: !Integer,
    fcSqrt2 :: !RealT,
    fcAmplitudeByWaist :: !RealT
  }
  deriving (Show)

integrateTrajectories :: (Vector Position, Vector Momentum) -> (Vector Position, Vector Momentum)
integrateTrajectories (initialPositions, initialMomenta) =
  let variables = VU.zip initialPositions initialMomenta
      !fieldConsts = precomputeFieldConstants
      n = VU.length variables
      chunkSize = max 1 ((n + numCapabilities - 1) `div` numCapabilities)
      chunks = chunksOf chunkSize variables
      processedChunks =
        map (VU.map (integrateNSteps fieldConsts numIntegrationSteps)) chunks
          `using` parList rdeepseq
      finalVariables = VU.concat processedChunks
   in VU.unzip finalVariables

chunksOf :: (VU.Unbox a) => Int -> VU.Vector a -> [VU.Vector a]
chunksOf size v
  | VU.null v = []
  | otherwise = let (h, t) = VU.splitAt size v in h : chunksOf size t

precomputeFieldConstants :: FieldConstants
precomputeFieldConstants =
  let wn = 2 * pi / wavelength
      wrs = waistRadius * waistRadius
      rl = pi * wrs / wavelength
      absL = abs azimuthalIndex
      sq2 = sqrt 2
      abw = amplitude * waistRadius
   in FieldConstants wn wrs rl absL sq2 abw

integrateNSteps :: FieldConstants -> Int -> (Position, Momentum) -> (Position, Momentum)
integrateNSteps fc n0 !state0 = go n0 state0
  where
    go 0 !s = s
    go k !s = go (k - 1) (integrationStep fc s)

integrationStep :: FieldConstants -> (Position, Momentum) -> (Position, Momentum)
integrationStep fc (!position, !momentum) =
  let (Position positionVector) = position
      (V4 laboratoryTime x y z) = positionVector
      (e, b) = computeLaguerreGaussElectricAndMagneticField fc (V3 x y z) laboratoryTime
      coeff = cutOff (laboratoryTime - z / c)
      V3 ex ey ez = e
      V3 bx by bz = b
      !eMod = V3 (coeff * ex) (coeff * ey) (coeff * ez)
      !bMod = V3 (coeff * bx) (coeff * by) (coeff * bz)
      (Momentum momentumVector) = momentum
      acceleration = computeAcceleration momentumVector eMod bMod
      V4 gamma px py pz = momentumVector
      V4 agamma ax ay az = acceleration
      !newMomentumVector =
        V4
          (gamma + integrationTimeStep * agamma)
          (px + integrationTimeStep * ax)
          (py + integrationTimeStep * ay)
          (pz + integrationTimeStep * az)
      V4 newGamma newPx newPy newPz = newMomentumVector
      !newPositionVector =
        V4
          (laboratoryTime + integrationTimeStep * newGamma)
          (x + integrationTimeStep * newPx)
          (y + integrationTimeStep * newPy)
          (z + integrationTimeStep * newPz)
   in (Position newPositionVector, Momentum newMomentumVector)

cutOff :: RealT -> RealT
cutOff !phi = exp (-t ^ (2 :: Int))
  where
    t = (phi - phi0) / tau0

tolerance :: RealT
tolerance = 1e-8

computeLaguerreGaussElectricAndMagneticField :: FieldConstants -> Vec3 -> RealT -> (Vec3, Vec3)
computeLaguerreGaussElectricAndMagneticField !(FieldConstants wn _ rl absL sq2 abw) !(V3 x y z) time =
  let xx = x * x
      yy = y * y
      r = sqrt (xx + yy)
      phi = atan2 y x
      zOverRayleigh = z / rl
      width = waistRadius * sqrt (1 + zOverRayleigh * zOverRayleigh)
      rOverWidth = r / width
      rOverWidthSquared = rOverWidth * rOverWidth
      radiusOfCurvature =
        if abs z < tolerance
          then 0
          else
            let rayleighOverZ = rl / z
             in z * (1 + rayleighOverZ * rayleighOverZ)
      curvature =
        if abs radiusOfCurvature < tolerance
          then 0
          else (xx + yy) / (2 * radiusOfCurvature)
      gouyPhase = atan2 z rl

      polynomialPart = laguerrePolynomialSpecialized (fromIntegral absL) (2 * rOverWidthSquared)
      exponentialDecay = exp (-rOverWidthSquared)
      magnitude = abw * (1 / width) * (sq2 * rOverWidth) ^ absL * polynomialPart * exponentialDecay

      timePhaseShift = angularVelocity * time
      longitudinalPhaseShift = wn * z
      curvaturePhaseShift = wn * curvature
      azimuthalPhaseShift = (fromIntegral azimuthalIndex) * phi
      gouyPhaseShift = (2 * (fromIntegral radialIndex) + (fromIntegral absL) + 1) * gouyPhase
      totalPhaseShift = timePhaseShift - longitudinalPhaseShift + curvaturePhaseShift + azimuthalPhaseShift - gouyPhaseShift

      coeffReal = magnitude * cos totalPhaseShift
      coeffImag = magnitude * sin totalPhaseShift

      prx :+ pix = polarizationX
      pry :+ piy = polarizationY
      exReal = coeffReal * prx - coeffImag * pix
      exImag = coeffReal * pix + coeffImag * prx
      eyReal = coeffReal * pry - coeffImag * piy
      eyImag = coeffReal * piy + coeffImag * pry

      widthSquared = width * width
      ezReal = (-2) * (x * exImag + y * eyImag) / (wn * widthSquared)

      e = V3 exReal eyReal ezReal

      bxReal = (-eyReal) / c
      byReal = exReal / c
      bzReal = (-(y * exImag - x * eyImag)) / (angularVelocity * widthSquared)

      b = V3 bxReal byReal bzReal
   in (e, b)

laguerrePolynomialSpecialized :: RealT -> RealT -> RealT
laguerrePolynomialSpecialized !alpha !x =
  0.5 * (x * x - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2))

computeAcceleration :: Vec4 -> Vec3 -> Vec3 -> Vec4
computeAcceleration momentum e b =
  let V4 gamma px py pz = momentum
      V3 ex ey ez = e
      V3 bx by bz = b
      agamma = px * ex / c + py * ey / c + pz * ez / c
      ax = gamma * ex / c + py * bz - pz * by
      ay = gamma * ey / c - px * bz + pz * bx
      az = gamma * ez / c + px * by - py * bx
   in V4
        (chargeToMassRatio * agamma)
        (chargeToMassRatio * ax)
        (chargeToMassRatio * ay)
        (chargeToMassRatio * az)
