module ElectrodynamicsSimulations.InitialConditions (generateInitialConditions) where

import Control.Monad.Random (MonadRandom (getRandomR), Rand, RandomGen)
import Data.Massiv.Array qualified as A
import ElectrodynamicsSimulations.Constants
  ( diskRadius,
    numParticles,
  )
import ElectrodynamicsSimulations.Types (InitialConditions, Momentum (Momentum), Position (Position))
import Linear (V4 (V4))

generateRandomPosition :: (RandomGen g) => Rand g Position
generateRandomPosition = do
  let diskRadiusSquared = diskRadius ^ (2 :: Int)

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

generateInitialPositions :: (RandomGen g) => Int -> Rand g (A.Array A.S A.Ix1 Position)
generateInitialPositions n =
  A.makeArrayAR A.S (A.Sz1 n) $ const generateRandomPosition

generateInitialConditions :: (RandomGen g) => Rand g InitialConditions
generateInitialConditions = do
  let n = fromIntegral numParticles
  initialPositions <- generateInitialPositions n
  let initialMomenta = A.replicate A.Seq (A.Sz1 n) initialMomentum
  return (initialPositions, initialMomenta)
