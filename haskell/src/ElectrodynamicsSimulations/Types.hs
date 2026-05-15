{-# LANGUAGE TypeFamilies #-}

module ElectrodynamicsSimulations.Types
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

import Control.DeepSeq (NFData)
import Data.Binary (Binary (get, put))
import Data.Binary.Get (getDoublele)
import Data.Binary.Put (putDoublele)
import Data.Complex (Complex)
import Data.Vector.Generic qualified as VG
import Data.Vector.Generic.Mutable qualified as VGM
import Data.Vector.Unboxed (Vector)
import Data.Vector.Unboxed qualified as VU
import Foreign (Storable)
import Linear (V3, V4 (V4))

type RealT = Double

type ComplexT = Complex RealT

type Vec3 = V3 RealT

type Vec4 = V4 RealT

newtype Position = Position Vec4
  deriving (Show, Storable, NFData)

newtype instance VU.MVector s Position = MV_Position (VU.MVector s Vec4)

newtype instance VU.Vector Position = V_Position (VU.Vector Vec4)

deriving instance VGM.MVector VU.MVector Position

deriving instance VG.Vector VU.Vector Position

instance VU.Unbox Position

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

newtype instance VU.MVector s Momentum = MV_Momentum (VU.MVector s Vec4)

newtype instance VU.Vector Momentum = V_Momentum (VU.Vector Vec4)

deriving instance VGM.MVector VU.MVector Momentum

deriving instance VG.Vector VU.Vector Momentum

instance VU.Unbox Momentum

type InitialConditions = (Vector Position, Vector Momentum)

newtype AngularMomentum = AngularMomentum RealT
  deriving (Show, Storable, NFData)

newtype instance VU.MVector s AngularMomentum = MV_AngularMomentum (VU.MVector s RealT)

newtype instance VU.Vector AngularMomentum = V_AngularMomentum (VU.Vector RealT)

deriving instance VGM.MVector VU.MVector AngularMomentum

deriving instance VG.Vector VU.Vector AngularMomentum

instance VU.Unbox AngularMomentum

instance Binary AngularMomentum where
  put (AngularMomentum am) = putDoublele am

  get = do
    am <- getDoublele
    return $ AngularMomentum am
