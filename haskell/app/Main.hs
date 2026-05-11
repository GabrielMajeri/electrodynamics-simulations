module Main (main) where

import Constants (numParticles, numIntegrationSteps, integrationStartTime, integrationEndTime,integrationTimeStep)
import Data.Binary.Put (runPut)
import Data.ByteString.Lazy (hPut)
import Lib ( generateInitialConditions, integrateTrajectories, computeAngularMomentaInZDirection )
import NPY (toNPYFile)
import System.CPUTime ( getCPUTime )
import System.IO ( IOMode(WriteMode), openFile, hClose )
import System.Random (mkStdGen)
import Text.Printf
import Types (Position (Position), InitialConditions)
import Control.Monad.Random (evalRand)
import Control.DeepSeq (rnf)

randomSeed :: Int
randomSeed = 42

main :: IO ()
main = do
  (initialPositions, initialMomenta) <- initialConditionsGeneration

  saveInitialPositionsToDisk initialPositions

  putStrLn ""

  printf "Starting the numerical integration of %d particle trajectories\n" numParticles
  printf "Integrating from initial time = %f to final time = %f with a time step of %f, a total of %d time steps\n" integrationStartTime integrationEndTime integrationTimeStep numIntegrationSteps

  start <- seq (rnf initialPositions, rnf initialMomenta) getCPUTime
  let (finalPositions, finalMomenta) = integrateTrajectories (initialPositions, initialMomenta)
  end <- seq (rnf finalPositions, rnf finalMomenta) getCPUTime
  let duration = (fromIntegral (end - start)) / (10 ^ 12)
  printf "Integrating trajectories took %0.6f seconds\n" (duration :: Double)

  putStrLn ""

  start <- seq finalPositions getCPUTime
  let angularMomenta = computeAngularMomentaInZDirection finalPositions finalMomenta
  end <- seq angularMomenta getCPUTime
  let duration = (fromIntegral (end - start)) / (10 ^ 12)
  printf "Computing angular momenta took %0.6f seconds\n" (duration :: Double)

  putStrLn "Saving final angular momenta to disk..."
  file <- openFile "angular_momenta.npy" WriteMode
  hPut file $ runPut $ toNPYFile angularMomenta
  hClose file

  putStrLn "Done!"

initialConditionsGeneration :: IO InitialConditions
initialConditionsGeneration = do
  let g = mkStdGen randomSeed

  printf "Starting to generate initial conditions for %d particles...\n" numParticles
  start <- getCPUTime

  let initialConditions = evalRand generateInitialConditions g

  end <- seq (rnf initialConditions) getCPUTime
  let duration = (fromIntegral (end - start)) / (10 ^ 12)
  printf "Generating %d initial conditions took %0.6f seconds\n" numParticles (duration :: Double)

  return initialConditions

saveInitialPositionsToDisk :: [Position] -> IO ()
saveInitialPositionsToDisk initialPositions = do
  putStrLn "Saving initial positions to disk..."
  start <- seq (rnf initialPositions) getCPUTime

  file <- openFile "initial_positions.npy" WriteMode
  hPut file $ runPut $ toNPYFile initialPositions
  hClose file

  end <- getCPUTime
  let duration = (fromIntegral (end - start)) / (10 ^ 12)
  printf "Dumping initial positions to disk took %0.6f seconds\n" (duration :: Double)
