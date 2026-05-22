{-# LANGUAGE TypeFamilies #-}

module ElectrodynamicsSimulations.Types
  ( RealT,
    ComplexT,
    Vec3,
    Vec4,
    SimArray,
    Position (Position),
    Momentum (Momentum),
    InitialConditions,
    AngularMomentum (AngularMomentum),
  )
where

import Control.DeepSeq (NFData)
import Data.Binary (Binary (get, put))
import Data.Binary.Get (getDoublele)
import Data.Binary.Put (putDoublele)
import Data.Complex (Complex)
import Data.Massiv.Array qualified as A
import Foreign (Storable)
import Linear (V3, V4 (V4))

type RealT = Double

type ComplexT = Complex RealT

type Vec3 = V3 RealT

type Vec4 = V4 RealT

newtype Position = Position Vec4
  deriving (Show, Storable, NFData)

instance Binary Position where
  put (Position (V4 a x y z)) = do
    putDoublele a
    putDoublele x
    putDoublele y
    putDoublele z

  get = do
    a <- getDoublele
    x <- getDoublele
    y <- getDoublele
    z <- getDoublele
    return $ Position (V4 a x y z)

newtype Momentum = Momentum Vec4
  deriving (Show, Storable, NFData)

type SimArray a = A.Array A.S A.Ix1 a

type InitialConditions = (SimArray Position, SimArray Momentum)

newtype AngularMomentum = AngularMomentum RealT
  deriving (Show, Storable, NFData)

instance Binary AngularMomentum where
  put (AngularMomentum am) = putDoublele am

  get = do
    am <- getDoublele
    return $ AngularMomentum am
