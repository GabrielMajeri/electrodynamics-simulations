module ElectrodynamicsSimulations.NPY (toNPYFile) where

import Data.Binary (Binary (put), Word8)
import Data.Binary.Put (Put, putLazyByteString, putShortByteString, putWord32le, putWord8, runPut)
import Data.ByteString.Lazy qualified as BL
import Data.Char (ord)
import Data.String (IsString (fromString))
import Data.Vector.Unboxed (Unbox, Vector, toList, (!))
import Data.Vector.Unboxed qualified as VU
import Foreign (Storable (sizeOf))
import ElectrodynamicsSimulations.Types (RealT)

serializeMagicHeader :: Put
serializeMagicHeader = do
  -- Magic number
  putWord8 0x93
  putShortByteString $ fromString "NUMPY"
  -- Version 3.0
  putWord8 0x03
  putWord8 0x00

singleQuote :: Word8
singleQuote = fromIntegral $ ord '\''

comma :: Word8
comma = fromIntegral $ ord ','

space :: Word8
space = fromIntegral $ ord ' '

newline :: Word8
newline = fromIntegral $ ord '\n'

serializeDictionary :: (Storable a, Unbox a) => Vector a -> Put
serializeDictionary array = do
  -- DType descriptor
  putShortByteString $ fromString "{'descr':"
  putWord8 singleQuote
  let realSizeInBytes = sizeOf (0 :: RealT)
  putShortByteString $ fromString ("<f" ++ (show realSizeInBytes))
  putWord8 singleQuote
  putWord8 comma

  -- Fortran order (column major) flag
  putShortByteString $ fromString "'fortran_order':False"
  putWord8 comma

  -- Shape
  let elementSizeInBytes = sizeOf (array ! 0)
  let innerDimension :: Int = floor $ ((fromIntegral elementSizeInBytes) / (fromIntegral realSizeInBytes) :: Double)
  let shapeTupleString = "(" ++ (show $ VU.length array) ++ "," ++ (show innerDimension) ++ ")"
  putShortByteString $ fromString ("'shape':" ++ shapeTupleString)

  putWord8 $ fromIntegral $ ord '}'

dataSectionAlignment :: Int
dataSectionAlignment = 64

toNPYFile :: (Binary a, Storable a, Unbox a) => Vector a -> Put
toNPYFile !array = do
  let magicHeader = runPut $ serializeMagicHeader
  let magicHeaderBytes = BL.unpack magicHeader
  let magicHeaderSize = length magicHeaderBytes

  let dictionary = runPut $ serializeDictionary array
  let dictionaryBytes = BL.unpack dictionary
  let dictionarySize = length dictionaryBytes

  let headerContentsSize = magicHeaderSize + 4 + dictionarySize
  let paddingSize = dataSectionAlignment - ((headerContentsSize + 1) `mod` dataSectionAlignment)
  let dictionaryWithPaddingSize = dictionarySize + paddingSize + 1

  putLazyByteString magicHeader
  putWord32le $ fromIntegral dictionaryWithPaddingSize
  putLazyByteString dictionary
  putLazyByteString $ BL.pack (replicate paddingSize space)
  putWord8 newline

  mapM_ put (toList array)
