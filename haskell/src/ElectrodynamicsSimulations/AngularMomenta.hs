module ElectrodynamicsSimulations.AngularMomenta
  ( computeAngularMomentaInZDirection,
  )
where

import Data.Massiv.Array qualified as A
import ElectrodynamicsSimulations.Types (AngularMomentum (AngularMomentum), Momentum (Momentum), Position (Position), SimArray)
import Linear (V4 (V4))

computeAngularMomentumInZDirection :: Position -> Momentum -> AngularMomentum
computeAngularMomentumInZDirection (Position (V4 _ x y _)) (Momentum (V4 _ vx vy _)) =
  AngularMomentum $ x * vy - y * vx

computeAngularMomentaInZDirection :: (SimArray Position, SimArray Momentum) -> SimArray AngularMomentum
computeAngularMomentaInZDirection (positions, momenta) =
  A.computeAs A.S
    $ A.setComp A.Par
    $ A.zipWith computeAngularMomentumInZDirection positions momenta
