module Main (main) where

import Constants (numParticles)
import Data.Binary.Put (runPut)
import Data.ByteString.Lazy (hPut)
import Lib
import NPY (toNPYFile)
import System.CPUTime
import System.IO
import System.Random (mkStdGen)
import Text.Printf
import Types (Position (Position))
import Control.Monad.Random (evalRand)

randomSeed :: Int
randomSeed = 42

main :: IO ()
main = do
  let g = mkStdGen randomSeed
  start <- getCPUTime
  let initialConditions = evalRand generateInitialConditions g
  end <- getCPUTime
  let duration = (fromIntegral (end - start)) / (10 ^ 12)
  printf "Generating %d initial conditions took %0.6f seconds\n" numParticles (duration :: Double)
  let (initialPositions, initialMomenta) = initialConditions

  file <- openFile "initial_positions.npy" WriteMode
  hPut file $ runPut (toNPYFile (map (\(Position p) -> p) initialPositions))
  hClose file

  let (finalPositions, finalMomenta) = integrateTrajectories initialPositions initialMomenta

  print (head $ initialPositions)
  print (last $ initialPositions)
