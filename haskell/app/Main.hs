module Main (main) where

import Constants (integrationEndTime, integrationStartTime, integrationTimeStep, numIntegrationSteps, numParticles)
import Control.DeepSeq (deepseq)
import Control.Monad.Random (evalRand)
import Data.Binary.Put (runPut)
import Data.ByteString.Lazy (hPut)
import Data.Vector.Unboxed (Vector)
import GHC.Clock (getMonotonicTime)
import InitialConditions (generateInitialConditions)
import Lib (computeAngularMomentaInZDirection, integrateTrajectories)
import NPY (toNPYFile)
import System.IO (IOMode (WriteMode), hClose, openFile)
import System.Random (mkStdGen)
import Text.Printf (printf)
import Types (AngularMomentum, InitialConditions, Momentum, Position)

randomSeed :: Int
randomSeed = 42

main :: IO ()
main = do
  (initialPositions, initialMomenta) <- initialConditionsGeneration

  saveInitialPositionsToDisk initialPositions

  putStrLn ""

  finalState <- performTrajectoryIntegration (initialPositions, initialMomenta)

  putStrLn ""

  angularMomenta <- computeAngularMomentaForFinalState finalState

  saveAngularMomentaToDisk angularMomenta

  putStrLn "Done!"

initialConditionsGeneration :: IO InitialConditions
initialConditionsGeneration = do
  let g = mkStdGen randomSeed

  printf "Starting to generate initial conditions for %d particles...\n" numParticles
  start <- deepseq g getMonotonicTime

  let initialConditions = evalRand generateInitialConditions g

  end <- deepseq initialConditions getMonotonicTime
  let duration = (end - start)
  printf "Generating %d initial conditions took %0.6f seconds\n" numParticles duration

  return initialConditions

saveInitialPositionsToDisk :: Vector Position -> IO ()
saveInitialPositionsToDisk initialPositions = do
  putStrLn "Saving initial positions to disk..."
  start <- getMonotonicTime

  file <- openFile "initial_positions.npy" WriteMode
  hPut file $ runPut $ toNPYFile initialPositions
  hClose file

  end <- getMonotonicTime
  let duration = (end - start)
  printf "Saving initial positions to disk took %0.6f seconds\n" duration

performTrajectoryIntegration :: InitialConditions -> IO (Vector Position, Vector Momentum)
performTrajectoryIntegration initialState = do
  printf "Starting the numerical integration of %d particle trajectories\n" numParticles
  printf "Integrating from initial time = %f to final time = %f with a time step of %f, a total of %d time steps\n" integrationStartTime integrationEndTime integrationTimeStep numIntegrationSteps

  start <- deepseq initialState getMonotonicTime
  let finalState = integrateTrajectories initialState
  end <- deepseq finalState getMonotonicTime
  let duration = (end - start)
  printf "Integrating trajectories took %0.6f seconds\n" duration

  return finalState

computeAngularMomentaForFinalState :: (Vector Position, Vector Momentum) -> IO (Vector AngularMomentum)
computeAngularMomentaForFinalState initialState = do
  start <- deepseq initialState getMonotonicTime

  let angularMomenta = computeAngularMomentaInZDirection initialState

  end <- deepseq angularMomenta getMonotonicTime
  let duration = (end - start)
  printf "Computing angular momenta took %0.6f seconds\n" duration

  return angularMomenta

saveAngularMomentaToDisk :: Vector AngularMomentum -> IO ()
saveAngularMomentaToDisk angularMomenta = do
  putStrLn "Saving final angular momenta to disk..."

  start <- deepseq angularMomenta getMonotonicTime

  file <- openFile "angular_momenta.npy" WriteMode
  hPut file $ runPut $ toNPYFile angularMomenta
  hClose file

  end <- getMonotonicTime
  let duration = (end - start)
  printf "Saving angular momenta to disk took %0.6f seconds\n" duration
