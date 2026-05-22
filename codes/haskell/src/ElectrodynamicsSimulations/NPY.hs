module ElectrodynamicsSimulations.NPY (toNPYFile) where

import Data.Binary (Binary (put), Word8)
import Data.Binary.Put (Put, putLazyByteString, putShortByteString, putWord32le, putWord8, runPut)
import Data.ByteString.Lazy qualified as BL
import Data.Char (ord)
import Data.Massiv.Array qualified as A
import Data.String (IsString (fromString))
import ElectrodynamicsSimulations.Types (RealT, SimArray)
import Foreign (Storable (sizeOf))

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

serializeDictionary :: (Storable a) => SimArray a -> Put
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
  let numElements = A.elemsCount array
  let elementSizeInBytes =
        if numElements == 0
          then 0
          else sizeOf (array A.! A.Ix1 0)
  let innerDimension = elementSizeInBytes `quot` realSizeInBytes
  let shapeTupleString = "(" ++ (show numElements) ++ "," ++ (show innerDimension) ++ ")"
  putShortByteString $ fromString ("'shape':" ++ shapeTupleString)

  putWord8 $ fromIntegral $ ord '}'

dataSectionAlignment :: Int
dataSectionAlignment = 64

toNPYFile :: (Binary a, Storable a) => SimArray a -> Put
toNPYFile !array = do
  let magicHeader = runPut $ serializeMagicHeader
  let magicHeaderSize = fromIntegral (BL.length magicHeader)

  let dictionary = runPut $ serializeDictionary array
  let dictionarySize = fromIntegral (BL.length dictionary)

  let headerContentsSize = magicHeaderSize + 4 + dictionarySize
  let paddingSize = dataSectionAlignment - ((headerContentsSize + 1) `mod` dataSectionAlignment)
  let dictionaryWithPaddingSize = dictionarySize + paddingSize + 1

  putLazyByteString magicHeader
  putWord32le $ fromIntegral dictionaryWithPaddingSize
  putLazyByteString dictionary
  putLazyByteString $ BL.pack (replicate paddingSize space)
  putWord8 newline

  A.mapM_ put array
