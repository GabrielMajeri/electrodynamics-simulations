module Types (RealT, Vec3 (Vec3), Vec4 (Vec4), Position (Position), Momentum (Momentum)) where

import Data.Binary (Binary (put, get))
import Data.Binary.Put (putWord64le)
import Data.Binary.Get (getWord64le)
import Data.Bits.Floating (coerceToFloat, coerceToWord)

type RealT = Double

data Vec3 = Vec3 {x :: RealT, y :: RealT, z :: RealT}
  deriving (Show)

data Vec4 = Vec4 {a :: RealT, x :: RealT, y :: RealT, z :: RealT}
  deriving (Show)

instance Binary Vec4 where
    put (Vec4 a x y z) = do
        putWord64le $ coerceToWord a
        putWord64le $ coerceToWord x
        putWord64le $ coerceToWord y
        putWord64le $ coerceToWord z

    get = do
        a <- fmap coerceToFloat getWord64le
        x <- fmap coerceToFloat getWord64le
        y <- fmap coerceToFloat getWord64le
        z <- fmap coerceToFloat getWord64le
        return $ Vec4 a x y z


newtype Position = Position Vec4
  deriving (Show)

newtype Momentum = Momentum Vec4
  deriving (Show)
