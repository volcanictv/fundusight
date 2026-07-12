---
name: Clinical Liquid Glass
colors:
  surface: '#f7f9fb'
  surface-dim: '#d8dadc'
  surface-bright: '#f7f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f6'
  surface-container: '#eceef0'
  surface-container-high: '#e6e8ea'
  surface-container-highest: '#e0e3e5'
  on-surface: '#191c1e'
  on-surface-variant: '#464555'
  inverse-surface: '#2d3133'
  inverse-on-surface: '#eff1f3'
  outline: '#777587'
  outline-variant: '#c7c4d8'
  surface-tint: '#4d44e3'
  primary: '#3525cd'
  on-primary: '#ffffff'
  primary-container: '#4f46e5'
  on-primary-container: '#dad7ff'
  inverse-primary: '#c3c0ff'
  secondary: '#006591'
  on-secondary: '#ffffff'
  secondary-container: '#39b8fd'
  on-secondary-container: '#004666'
  tertiary: '#7e3000'
  on-tertiary: '#ffffff'
  tertiary-container: '#a44100'
  on-tertiary-container: '#ffd2be'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#0f0069'
  on-primary-fixed-variant: '#3323cc'
  secondary-fixed: '#c9e6ff'
  secondary-fixed-dim: '#89ceff'
  on-secondary-fixed: '#001e2f'
  on-secondary-fixed-variant: '#004c6e'
  tertiary-fixed: '#ffdbcc'
  tertiary-fixed-dim: '#ffb695'
  on-tertiary-fixed: '#351000'
  on-tertiary-fixed-variant: '#7b2f00'
  background: '#f7f9fb'
  on-background: '#191c1e'
  surface-variant: '#e0e3e5'
typography:
  headline-lg:
    fontFamily: Manrope
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Manrope
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Manrope
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 20px
  margin-mobile: 16px
  margin-desktop: 40px
---

## Brand & Style

This design system embodies a high-fidelity, medical-grade aesthetic characterized by clarity, precision, and a "sanitized" digital atmosphere. It targets healthcare professionals and researchers who require data-dense environments that remain visually breathable.

The style is a refined **Light-Mode Glassmorphism**. It utilizes a frost-white foundation to evoke a sense of sterile efficiency and trust. The interface feels like layered sheets of polished acrylic or frosted glass, using subtle back-layer blurs and micro-shadows to define hierarchy without heavy visual weight. The emotional response is one of calm, professional focus and technical sophistication.

## Colors

The palette is anchored by **Frost White (#F7F9FB)**, a high-brightness base that serves as the "sterile" canvas. 

- **Primary:** Indigo (#4F46E5) is reserved exclusively for primary calls to action, active states, and critical data points.
- **Secondary:** A bright Sky Blue (#0EA5E9) is used for informational accents and secondary interactive elements.
- **Surface:** Surfaces use semi-transparent whites with a high background blur (20px-40px) to maintain the liquid glass effect.
- **Text:** Deep Slate (#1E293B) provides high legibility against the light, translucent backgrounds.

## Typography

Typography prioritizes systematic clarity. **Manrope** provides a modern, balanced feel for headings, suggesting technical precision. **Inter** is used for body copy to ensure maximum readability in dense data contexts. **JetBrains Mono** is utilized for labels, technical values, and status indicators to emphasize the "clinical instrument" nature of the product. 

On mobile devices, headline sizes are scaled down to maintain hierarchy without overwhelming the limited screen real estate.

## Layout & Spacing

The design system employs a **Fluid Grid** model with high-margin padding to preserve the sense of openness. 

- **Desktop:** A 12-column grid with 20px gutters and generous 40px outer margins.
- **Tablet:** An 8-column grid with 16px gutters.
- **Mobile:** A 4-column grid with 16px gutters and 16px margins.

Spacing follows a strict 4px baseline rhythm. Horizontal and vertical padding within glass containers should be generous (typically `md` or `lg`) to prevent the content from feeling "compressed" against the translucent edges.

## Elevation & Depth

Depth is achieved through **Glassmorphism and Ambient Shadows**. Rather than using dark shadows, this system uses "Light Diffusion":

1.  **Backdrop Filter:** All elevated surfaces must apply a `blur(20px)` to the layer beneath them.
2.  **Surface Tint:** Surfaces use a 70% opacity white fill.
3.  **Refractive Border:** A 1px solid white border at 40% opacity is used to simulate the edge of a glass pane.
4.  **Soft Shadow:** A very large, very soft shadow (e.g., `0 10px 30px rgba(0, 0, 0, 0.04)`) provides a subtle "lift" without appearing heavy or dirty on the frost-white background.

## Shapes

The shape language is **Rounded**. This softens the clinical nature of the product, making it feel more approachable and ergonomic. 

Standard components (inputs, small buttons) use a 0.5rem radius. Larger layout containers and cards use a 1rem radius (rounded-lg) to emphasize the "sheet of glass" metaphor. This consistent curvature mimics high-end medical hardware.

## Components

- **Buttons:** Primary buttons are solid Indigo (#4F46E5) with white text. Secondary buttons use the Glassmorphism style: a translucent white background with an Indigo border and text.
- **Input Fields:** Fields are slightly recessed with a soft inner-shadow or a subtle 1px border (#E2E8F0). When focused, they gain a 2px Indigo glow.
- **Cards:** These are the primary vessels for information. They feature the signature background blur, white translucent fill, and a subtle 1px white border.
- **Chips/Status:** Use low-saturation background tints (e.g., light green for "Stable") with high-saturation text to maintain the clinical, clean look.
- **Lists:** Items are separated by ultra-thin, low-opacity lines. Hover states on list items use a slight increase in background opacity (to 90% white) rather than a color change.
- **Data Visualization:** Charts should use the primary Indigo and secondary Sky Blue, using semi-transparent gradients to fill areas, maintaining the "liquid" theme.