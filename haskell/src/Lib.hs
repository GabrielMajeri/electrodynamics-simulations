module Lib
  ( generateInitialConditions,
    integrateTrajectories,
  )
where

import Constants (diskRadius, numParticles)
import Types (Momentum (Momentum), Position (Position), Vec4 (Vec4))
import Control.Monad.Random (Rand, RandomGen, MonadRandom (getRandomR))

generateRandomPosition :: RandomGen g => Rand g Position
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

  return $ Position $ Vec4 0 x y 0

initialMomentum :: Momentum
initialMomentum = Momentum $ Vec4 1 0 0 0

generateInitialPositions :: RandomGen g => Int -> Rand g [Position]
generateInitialPositions n = sequence $ replicate n generateRandomPosition

generateInitialConditions :: RandomGen g => Rand g ([Position], [Momentum])
generateInitialConditions = do
  let n = fromIntegral numParticles
  initialPositions <- generateInitialPositions n
  let initialMomenta = replicate n initialMomentum
  return $ (initialPositions, initialMomenta)

integrateTrajectories :: [Position] -> [Momentum] -> ([Position], [Momentum])
integrateTrajectories initialPositions initialMomenta = undefined
