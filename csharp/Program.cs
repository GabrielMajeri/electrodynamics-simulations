using Real = double;
using Complex = System.Numerics.Complex;

Console.WriteLine("Simulation of electrons accelerated by a Laguerre-Gauss beam");

const uint NumElectrons = 64 * 1024;
const int Seed = 17;

const uint RadialIndex = 2;
const int AzimuthalIndex = -2;

const Real c = 137.036;
const Real Omega = 0.057;
const Real Wavelength = 2 * Math.PI * c / Omega;
const Real WaistRadius = 75 * Wavelength;
const Real DiskRadius = (1.75 + RadialIndex) * WaistRadius;

const Real A0 = 1e-2;
const Real ElectronMass = 1;
const Real ElectronCharge = -1;
Real Amplitude = A0 * ElectronMass * c * Omega / Math.Abs(ElectronCharge);

Console.WriteLine("Generating initial positions for {0} electrons", NumElectrons);

var watch = System.Diagnostics.Stopwatch.StartNew();
var initialPositions = GenerateInitialElectronPositions(NumElectrons, DiskRadius, Seed);
watch.Stop();
Console.WriteLine("Took {0} seconds to generate initial positions", watch.Elapsed.TotalSeconds);

var initialMomenta = GenerateInitialElectronMomenta(NumElectrons);

const Real Tau0 = 10 / Omega;
const Real Phi0 = 3 * Tau0;
const Real StartTime = 0;
const Real EndTime = 6 * Tau0;
const Real TimeStep = Tau0 / 2048;

var parameters = new LaguerreGaussBeamParameters
{
    Amplitude = Amplitude,
    Polarization = PolarizationVector.Linear,
    WaistRadius = WaistRadius,
    Wavelength = Wavelength,
    AngularVelocity = Omega,
    RadialIndex = RadialIndex,
    AzimuthalIndex = AzimuthalIndex,
};

Console.WriteLine("Integrating equations of motions from t_0 = {0} up to t_final = {1}, with a time step of {2}", StartTime, EndTime, TimeStep);

watch = System.Diagnostics.Stopwatch.StartNew();

var (finalPositions, finalMomenta) = IntegrateTrajectories(parameters, initialPositions, initialMomenta, StartTime, EndTime, TimeStep);

watch.Stop();

Console.WriteLine("Took {0} seconds to integrate trajectories", watch.Elapsed.TotalSeconds);

static Position[] GenerateInitialElectronPositions(uint numElectrons, double diskRadius, int seed)
{
    var positions = new Position[numElectrons];

    var rand = new Random(seed);

    // Use the inverse sampling method to generate points uniformly in the disk
    var DiskRadiusSquared = DiskRadius * DiskRadius;

    for (var index = 0; index < numElectrons; ++index)
    {
        var r = Math.Sqrt(rand.NextDouble() * DiskRadiusSquared);
        var theta = rand.NextDouble() * (2 * Math.PI);

        var (sinTheta, cosTheta) = Math.SinCos(theta);
        positions[index] = new Position { t = 0, x = r * cosTheta, y = r * sinTheta, z = 0 };
    }

    return positions;
}

static Momentum[] GenerateInitialElectronMomenta(uint numElectrons)
{
    var momenta = new Momentum[numElectrons];

    for (var index = 0; index < numElectrons; ++index)
    {
        momenta[index] = new Momentum { gamma = 1, vx = 0, vy = 0, vz = 0 };
    }

    return momenta;
}

static (Position[], Momentum[]) IntegrateTrajectories(
    LaguerreGaussBeamParameters parameters,
    Position[] initialPositions, Momentum[] initialMomenta,
    Real integrationStartTime, Real integrationEndTime, Real timeStep
)
{
    int numParticles = initialPositions.Length;
    if (initialMomenta.Length != numParticles)
    {
        throw new ArgumentException("Must have same number of initial position and initial momenta");
    }

    var positions = (Position[])initialPositions.Clone();
    var momenta = (Momentum[])initialMomenta.Clone();

    var integrationDuration = integrationEndTime - integrationStartTime;
    var numSteps = (uint)Math.Floor(integrationDuration / timeStep);

    // for (var index = 0; index < numParticles; ++index)
    DotMP.Parallel.ParallelFor(0, numParticles, index =>
    {
        Real currentTime = 0;
        for (var step = 0; step <= numSteps; ++step)
        {
            var previousPosition = positions[index];
            var laboratoryTime = previousPosition.t;
            var positionVector = new Vector3D { x = previousPosition.x, y = previousPosition.y, z = previousPosition.z };

            // Compute EM field vectors for previous position
            var (electricField, magneticField) =
                ComputeElectricAndMagenticFieldsForLaguerreGaussBeam(
                    parameters, positionVector, laboratoryTime
                );

            var cutOff = CutOff(laboratoryTime - previousPosition.z / c, Phi0, Tau0);
            electricField = cutOff * electricField;
            magneticField = cutOff * magneticField;

            var previousMomentum = momenta[index];

            // Symplectic Euler integration step
            var acceleration = ComputeElectromagneticAcceleration(
                previousMomentum, electricField, magneticField);

            var newMomentum = previousMomentum + timeStep * acceleration;
            var newPosition = previousPosition + timeStep * newMomentum;

            positions[index] = newPosition;
            momenta[index] = newMomentum;
        }

        currentTime += timeStep;
    });

    return (positions, momenta);
}

static (Vector3D, Vector3D) ComputeElectricAndMagenticFieldsForLaguerreGaussBeam(
    LaguerreGaussBeamParameters parameters,
    Vector3D position,
    Real time
)
{
    var r = Real.Hypot(position.x, position.y);
    var phi = Real.Atan2(position.y, position.x);
    var x = position.x;
    var y = position.y;
    var z = position.z;

    // z_R
    var rayleighLength = Math.PI * Math.Pow(parameters.WaistRadius, 2) / parameters.Wavelength;

    // w(z)
    var width = parameters.WaistRadius * Math.Sqrt(1 + Math.Pow(z / rayleighLength, 2));

    // r / w(z)
    var rOverWidth = r / width;
    var rOverWidthSquared = Math.Pow(rOverWidth, 2);

    // k
    var wavenumber = 2 * Math.PI / parameters.Wavelength;

    // |l|
    var AbsL = int.Abs(parameters.AzimuthalIndex);

    // R(z)
    var radiusOfCurvature = Real.Abs(z) < 1e-5 ? 0 : z * (1 + Math.Pow(rayleighLength / z, 2));

    // r^2/(2 * R(z))
    var curvaturePhaseTerm = radiusOfCurvature == 0 ? 0 : Math.Pow(r, 2) / (2 * radiusOfCurvature);

    // \psi(z)
    var gouyPhase = Math.Atan2(z, rayleighLength);

    var magnitude = parameters.Amplitude * (parameters.WaistRadius / width) * Math.Pow(Math.Sqrt(2) * rOverWidth, AbsL) * LaguerrePolynomial(parameters.RadialIndex, AbsL, 2 * rOverWidthSquared) * Math.Exp(-rOverWidthSquared);

    var phase = parameters.AngularVelocity * time - (wavenumber * z + wavenumber * curvaturePhaseTerm + parameters.AzimuthalIndex * phi - (2 * parameters.RadialIndex + AbsL + 1) * gouyPhase);
    var phaseMultiplier = Complex.Exp(Complex.ImaginaryOne * phase);

    var coeff = magnitude * phaseMultiplier;

    var widthSquared = Math.Pow(width, 2);

    var E_x = coeff * parameters.Polarization.x;
    var E_y = coeff * parameters.Polarization.y;
    var E_z = 2 * Complex.ImaginaryOne / (wavenumber * widthSquared) * (x * E_x + y * E_y);

    var E = new Vector3D { x = E_x.Real, y = E_y.Real, z = E_z.Real };

    var B_x = -E_y / c;
    var B_y = E_x / c;
    var B_z = Complex.ImaginaryOne / (parameters.AngularVelocity * widthSquared) * (y * E_x - x * E_y);

    var B = new Vector3D { x = B_x.Real, y = B_y.Real, z = B_z.Real };

    return (E, B);
}


// Laguerre gauss polynomial evaluation (fast for small n).
static Real LaguerrePolynomial(uint n, Real alpha, Real x)
{
    if (n == 0)
    {
        return 1;
    }

    if (n == 1)
    {
        return 1 + alpha - x;
    }

    if (n == 2)
    {
        return 0.5 * (x * x - 2 * (alpha + 2) * x + (alpha + 1) * (alpha + 2));
    }

    return ((2 * n - 1 + alpha - x) * LaguerrePolynomial(n - 1, alpha, x) - (n - 1 + alpha) * LaguerrePolynomial(n - 2, alpha, x)) / n;
}

static Real CutOff(Real phi, Real phi0, Real tau0)
{
    var argument = (phi - phi0) / tau0;
    return Real.Exp(-argument * argument);
}


static Acceleration ComputeElectromagneticAcceleration(
    Momentum previousMomentum,
    Vector3D electricField, Vector3D magneticField
)
{
    const Real ChargeToMassRatio = ElectronCharge / ElectronMass;

    var agamma = previousMomentum.vx * electricField.x / c + previousMomentum.vy * electricField.y / c + previousMomentum.vz * electricField.z / c;
    var ax = previousMomentum.gamma * electricField.x / c + previousMomentum.vy * magneticField.z - previousMomentum.vz * magneticField.y;
    var ay = previousMomentum.gamma * electricField.y / c - previousMomentum.vx * magneticField.z + previousMomentum.vz * magneticField.x;
    var az = previousMomentum.gamma * electricField.z / c - previousMomentum.vx * magneticField.y - previousMomentum.vy * magneticField.x;

    return ChargeToMassRatio * new Acceleration { dgamma = agamma, dvx = ax, dvy = ay, dvz = az };
}

struct Position
{
    public Real t, x, y, z;

    public static Position operator +(Position p, Momentum m)
    {
        return new Position
        {
            t = p.t + m.gamma,
            x = p.x + m.vx,
            y = p.y + m.vy,
            z = p.z + m.vz,
        };
    }
}

struct Vector3D
{
    public Real x, y, z;

    public static Vector3D operator *(Real scalar, Vector3D vector)
    {
        return new Vector3D { x = scalar * vector.x, y = scalar * vector.y, z = scalar * vector.z };
    }
}

struct Momentum
{
    public Real gamma, vx, vy, vz;

    public static Momentum operator +(Momentum m, Acceleration acc)
    {
        return new Momentum
        {
            gamma = m.gamma + acc.dgamma,
            vx = m.vx + acc.dvx,
            vy = m.vy + acc.dvy,
            vz = m.vz + acc.dvz,
        };
    }

    public static Momentum operator *(Real scalar, Momentum m)
    {
        return new Momentum
        {
            gamma = scalar * m.gamma,
            vx = scalar * m.vx,
            vy = scalar * m.vy,
            vz = scalar * m.vz,
        };
    }
}

struct Acceleration
{
    public Real dgamma, dvx, dvy, dvz;

    public static Acceleration operator *(Real scalar, Acceleration acc)
    {
        return new Acceleration
        {
            dgamma = scalar * acc.dgamma,
            dvx = scalar * acc.dvx,
            dvy = scalar * acc.dvy,
            dvz = scalar * acc.dvz,
        };
    }
}

struct PolarizationVector
{
    public Complex x, y;

    public static readonly PolarizationVector Linear = new() { x = 1, y = 0 };
    public static readonly PolarizationVector RightCircular = new() { x = 1 / Math.Sqrt(2), y = Complex.ImaginaryOne / Math.Sqrt(2) };
    public static readonly PolarizationVector LeftCircular = new() { x = 1 / Math.Sqrt(2), y = -Complex.ImaginaryOne / Math.Sqrt(2) };

}

struct LaguerreGaussBeamParameters
{
    // E_0 = B_0
    public Real Amplitude;

    // \xi
    public PolarizationVector Polarization;

    // w(0)
    public Real WaistRadius;

    // \lambda
    public Real Wavelength;

    // \omega
    public Real AngularVelocity;

    // p
    public uint RadialIndex;

    // l (or m)
    public int AzimuthalIndex;

};
