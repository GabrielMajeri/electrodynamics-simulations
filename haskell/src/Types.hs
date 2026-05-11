module Types
  ( RealT,
    ComplexT,
    Vec3,
    Vec4,
    Position (Position),
    Momentum (Momentum),
    InitialConditions,
    AngularMomentum (AngularMomentum),
  )
where

import Data.Complex (Complex)
import Linear (V3, V4 (V4))
import Data.Binary (Binary (get, put))
import Foreign (Storable)
import Data.Binary.Put (putDoublele)
import Data.Binary.Get (getDoublele)
import Control.DeepSeq (NFData)

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
  deriving (Show, NFData)

type InitialConditions = ([Position], [Momentum])

newtype AngularMomentum = AngularMomentum RealT
  deriving (Show, Storable, NFData)

instance Binary AngularMomentum where
  put (AngularMomentum am) = putDoublele am

  get = do
    am <- getDoublele
    return $ AngularMomentum am
