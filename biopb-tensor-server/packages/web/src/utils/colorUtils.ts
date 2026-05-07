/**
 * Pseudo color options for channel display.
 */
export const PRESET_COLORS = [
  { name: "Auto", value: "auto" },
  { name: "Grayscale", value: "gray" },
  { name: "Green", value: "green" },
  { name: "Red", value: "red" },
  { name: "Blue", value: "blue" },
  { name: "Magenta", value: "magenta" },
  { name: "Cyan", value: "cyan" },
  { name: "Yellow", value: "yellow" },
] as const;

export type PresetColor = (typeof PRESET_COLORS)[number]["value"];

/**
 * Color type can be either a preset named color or a hex color string (e.g., "#ff0000").
 * "auto" is a special value that means use the guessed color from channel name.
 */
export type ColorValue = PresetColor | string;

/**
 * Check if a color value is a hex color.
 */
export function isHexColor(color: ColorValue): boolean {
  return color.startsWith("#");
}

/**
 * RGB multipliers for each preset color.
 * Values are [R, G, B] normalized to range [0, 1].
 * Note: "auto" resolves to guessed color at render time.
 */
export const PRESET_COLOR_MULTIPLIERS: Record<PresetColor, [number, number, number]> = {
  auto: [1, 1, 1],  // Placeholder - will be resolved at render time
  gray: [1, 1, 1],
  green: [0, 1, 0],
  red: [1, 0, 0],
  blue: [0, 0, 1],
  magenta: [1, 0, 1],
  cyan: [0, 1, 1],
  yellow: [1, 1, 0],
};

/**
 * Convert a hex color string to RGB multipliers.
 * @param hex - Hex color string like "#ff0000" or "#f00"
 * @returns [R, G, B] normalized to range [0, 1]
 */
export function hexToMultipliers(hex: string): [number, number, number] {
  // Remove # prefix
  const cleanHex = hex.replace("#", "");

  // Expand shorthand (e.g., "f00" -> "ff0000")
  let fullHex = cleanHex;
  if (cleanHex.length === 3) {
    const c0 = cleanHex.charAt(0);
    const c1 = cleanHex.charAt(1);
    const c2 = cleanHex.charAt(2);
    fullHex = c0 + c0 + c1 + c1 + c2 + c2;
  }

  // Parse hex values
  const r = parseInt(fullHex.substring(0, 2), 16) / 255;
  const g = parseInt(fullHex.substring(2, 4), 16) / 255;
  const b = parseInt(fullHex.substring(4, 6), 16) / 255;

  return [r, g, b];
}

/**
 * Convert preset color name to a hex color for display.
 */
export const PRESET_TO_HEX: Record<PresetColor, string> = {
  auto: "#ffffff",  // Placeholder - will be resolved at display time
  gray: "#ffffff",
  green: "#00ff00",
  red: "#ff0000",
  blue: "#0000ff",
  magenta: "#ff00ff",
  cyan: "#00ffff",
  yellow: "#ffff00",
};

/**
 * Resolve "auto" color to the actual guessed color based on channel name.
 * If color is not "auto", returns it unchanged.
 */
export function resolveAutoColor(color: ColorValue, channelName?: string): ColorValue {
  if (color === "auto" && channelName) {
    return guessDefaultColor(channelName);
  }
  return color;
}

/**
 * Get RGB multipliers for a color value (preset or hex).
 * For "auto" color, pass channelName to resolve it.
 */
export function getColorMultipliers(color: ColorValue, channelName?: string): [number, number, number] {
  const resolvedColor = resolveAutoColor(color, channelName);
  if (isHexColor(resolvedColor)) {
    return hexToMultipliers(resolvedColor);
  }
  return PRESET_COLOR_MULTIPLIERS[resolvedColor as PresetColor] ?? [1, 1, 1];
}

/**
 * Get a hex color representation for display purposes.
 * For "auto" color, pass channelName to resolve it.
 */
export function colorToHex(color: ColorValue, channelName?: string): string {
  const resolvedColor = resolveAutoColor(color, channelName);
  if (isHexColor(resolvedColor)) {
    return resolvedColor;
  }
  return PRESET_TO_HEX[resolvedColor as PresetColor] ?? "#ffffff";
}

/**
 * Guess default display color from channel name.
 * Common fluorescence markers have conventional display colors.
 */
export function guessDefaultColor(channelName: string): PresetColor {
  const name = channelName.toLowerCase();

  // Green fluorescence markers
  if (/\bgfp\b|\bgreen\b|fitc|egfp|af488|cy2\b/.test(name)) return "green";

  // Red fluorescence markers
  if (/\brfp\b|\bred\b|mcherry|tritc|af568|af594|tdtomato|tomato/.test(name)) return "red";

  // Blue fluorescence markers (typically nuclear)
  if (/\bdapi\b|\bblue\b|hoechst|uv/.test(name)) return "blue";

  // Far red / magenta (Cy5 and similar)
  if (/\bcy5\b|\bmagenta\b|af647|af680|cy5\.5|cy7|ir800|ir700/.test(name)) return "magenta";

  // Cyan markers (Cy3 and similar)
  if (/\bcy3\b|\bcyan\b|\bcfp\b|cy3\.5/.test(name)) return "cyan";

  // Yellow markers
  if (/\byfp\b|\byellow\b|vfp/.test(name)) return "yellow";

  // Additional common patterns
  // Alexa Fluor series
  if (/af350|alexa350/.test(name)) return "blue";
  if (/af405|alexa405/.test(name)) return "blue";
  if (/af488|alexa488/.test(name)) return "green";
  if (/af555|alexa555/.test(name)) return "cyan";
  if (/af568|alexa568/.test(name)) return "red";
  if (/af594|alexa594/.test(name)) return "red";
  if (/af647|alexa647/.test(name)) return "magenta";
  if (/af680|alexa680/.test(name)) return "magenta";
  if (/af700|alexa700/.test(name)) return "magenta";
  if (/af750|alexa750/.test(name)) return "magenta";

  // Dye series
  if (/dyelight.*350|dl350/.test(name)) return "blue";
  if (/dyelight.*488|dl488/.test(name)) return "green";
  if (/dyelight.*550|dl550/.test(name)) return "cyan";
  if (/dyelight.*594|dl594/.test(name)) return "red";
  if (/dyelight.*633|dl633/.test(name)) return "magenta";
  if (/dyelight.*650|dl650/.test(name)) return "magenta";
  if (/dyelight.*680|dl680/.test(name)) return "magenta";
  if (/dyelight.*755|dl755/.test(name)) return "magenta";

  // Atto dyes
  if (/atto.*390|a390/.test(name)) return "blue";
  if (/atto.*425|a425/.test(name)) return "blue";
  if (/atto.*488|a488/.test(name)) return "green";
  if (/atto.*550|a550/.test(name)) return "cyan";
  if (/atto.*565|a565/.test(name)) return "red";
  if (/atto.*594|a594/.test(name)) return "red";
  if (/atto.*647|a647|atto647n/.test(name)) return "magenta";
  if (/atto.*680|a680/.test(name)) return "magenta";

  // Generic wavelength-based heuristics (extraction from channel name)
  const wavelengthMatch = name.match(/\b(\d{3})\b/);
  if (wavelengthMatch && wavelengthMatch[1]) {
    const wavelength = parseInt(wavelengthMatch[1], 10);
    if (wavelength >= 350 && wavelength <= 499) return "blue";
    if (wavelength >= 500 && wavelength <= 549) return "green";
    if (wavelength >= 550 && wavelength <= 569) return "cyan";
    if (wavelength >= 570 && wavelength <= 649) return "red";
    if (wavelength >= 650) return "magenta";
  }

  // Default to grayscale for unknown markers
  return "gray";
}

/**
 * Channel metadata structure for OME-NGFF (OME-Zarr).
 */
interface OmeroChannel {
  label?: string;
  name?: string;
  color?: string;
}

/**
 * Channel metadata structure for standard OME (OME-TIFF).
 */
interface OmeChannel {
  name?: string;
}

interface OmePixels {
  channels?: OmeChannel[];
}

interface OmeImage {
  pixels?: OmePixels;
}

interface MicroManagerSummary {
  ChNames?: string[];
}

interface Metadata {
  omero?: {
    channels?: OmeroChannel[];
  };
  images?: OmeImage[];
  Summary?: MicroManagerSummary;
}

/**
 * Extract channel names from metadata.
 * Supports OME-NGFF (omero.channels), standard OME (images[].pixels.channels),
 * and MicroManager (Summary.ChNames).
 */
export function extractChannelNames(
  metadata: Metadata,
  sceneId?: number,
): string[] {
  // Try OME-NGFF/Zarr format: omero.channels[].label or .name
  const omeroChannels = metadata.omero?.channels;
  if (omeroChannels && Array.isArray(omeroChannels)) {
    return omeroChannels.map((ch, i) => ch.label || ch.name || `Channel ${i}`);
  }

  // Try standard OME/OME-TIFF format: images[scene].pixels.channels[].name
  const images = metadata.images;
  if (Array.isArray(images) && images.length > 0) {
    const scene = images[sceneId ?? 0];
    if (scene?.pixels?.channels && Array.isArray(scene.pixels.channels)) {
      return scene.pixels.channels.map(
        (ch, i) => ch.name || `Channel ${i}`,
      );
    }
  }

  // Try MicroManager format: Summary.ChNames[]
  if (metadata.Summary?.ChNames && Array.isArray(metadata.Summary.ChNames)) {
    return metadata.Summary.ChNames.map((name, i) => name || `Channel ${i}`);
  }

  // No channel names found
  return [];
}