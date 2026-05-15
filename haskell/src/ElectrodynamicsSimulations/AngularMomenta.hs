module ElectrodynamicsSimulations.AngularMomenta
  ( computeAngularMomentaInZDirection,
  )
where

import ElectrodynamicsSimulations.Types (AngularMomentum (AngularMomentum), Momentum (Momentum), Position (Position))
import Data.Vector.Unboxed (Vector)
import qualified Data.Vector.Unboxed as VU
import Linear (V4 (V4))

computeAngularMomentumInZDirection :: Position -> Momentum -> AngularMomentum
computeAngularMomentumInZDirection (Position (V4 _ x y _)) (Momentum (V4 _ vx vy _)) =
  AngularMomentum $ x * vy - y * vx

computeAngularMomentaInZDirection :: (Vector Position, Vector Momentum) -> Vector AngularMomentum
computeAngularMomentaInZDirection (positions, momenta) =
  VU.zipWith computeAngularMomentumInZDirection positions momenta
