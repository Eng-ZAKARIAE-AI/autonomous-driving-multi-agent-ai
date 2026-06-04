# 1. Product Overview

## Product Concept

This product is a modern SaaS web application designed for professional teams who need to manage workflows, collaborate efficiently, track operational data, and make decisions from a clean, focused interface.

The interface should feel premium, fast, calm, and highly usable. It should combine the clarity of Notion, the precision of Linear, the visual polish of Stripe, and the refinement of Apple-style interaction design.

## Target Users

* Operations teams
* Product teams
* Business administrators
* Managers and team leads
* Power users who work with dashboards, records, forms, and workflows
* Mobile users who need quick access to key actions

## Core Product Experience

Users should be able to:

* Understand the product state at a glance
* Navigate between major areas quickly
* Create, edit, filter, and manage records
* Review insights and metrics
* Take contextual actions without friction
* Work comfortably in both light and dark mode
* Use the product efficiently across desktop, tablet, and mobile

## Product Personality

The product should feel:

* Modern
* Minimal
* Trustworthy
* Fast
* Premium
* Calm
* Professional
* Intelligent
* Highly polished

The design must avoid visual noise, excessive borders, low-contrast text, inconsistent spacing, and over-designed elements.

***

# 2. Design Principles

## Clarity First

Every screen should make the primary task obvious. Visual hierarchy must guide the user from high-level context to detailed actions.

Use clear headings, concise labels, grouped content, and predictable layouts.

## Calm Density

The UI should support information-rich workflows without feeling crowded.

Use spacing, subtle dividers, cards, and progressive disclosure to maintain readability.

## Reusable by Default

Every part of the interface should be composed from reusable components:

* Buttons
* Inputs
* Cards
* Tables
* Modals
* Tabs
* Side panels
* Empty states
* Status badges
* Dropdown menus

Avoid one-off visual patterns.

## Mobile-First Responsiveness

Design every feature so it works on small screens first, then progressively enhances on larger screens.

Mobile layouts should prioritize:

* Primary actions
* Search
* Filters
* Recently used content
* Clear tap targets
* Simplified navigation

## Accessible Premium

The product should feel beautiful without sacrificing accessibility.

All interactive states, text contrast, keyboard navigation, focus rings, and semantic structure must meet WCAG AA standards.

## Subtle Motion

Animation should support understanding, not decoration. Use soft transitions, small movements, and quick feedback.

## Consistent Hierarchy

Use consistent rules for:

* Page headers
* Section titles
* Cards
* Tables
* Forms
* Actions
* Navigation states

Users should never wonder whether two similar UI elements behave differently.

***

# 3. User Experience Goals

## Primary Goals

* Help users complete common tasks quickly
* Reduce cognitive load
* Make important information easy to scan
* Provide clear next steps on every screen
* Ensure the product feels fast and responsive
* Make complex workflows feel simple
* Support both beginner and power-user behavior

## Secondary Goals

* Encourage confidence through clear feedback
* Minimize unnecessary configuration
* Surface intelligent recommendations where helpful
* Support keyboard-first navigation on desktop
* Maintain visual consistency across all product areas

## UX Success Criteria

The interface is successful if users can:

* Identify where they are in the product within 2 seconds
* Find the primary action on each page immediately
* Complete form-based tasks without confusion
* Understand empty, loading, and error states clearly
* Navigate comfortably without reading documentation
* Use the interface equally well in light and dark mode

## Emotional Goals

Users should feel:

* In control
* Focused
* Confident
* Efficient
* Supported
* Uninterrupted

***

# 4. Information Architecture

## Global Structure

The application should use a dashboard-style SaaS architecture with persistent navigation and contextual content areas.

Recommended top-level areas:

1. Dashboard
2. Records
3. Projects
4. Tasks
5. Analytics
6. Team
7. Notifications
8. Settings

## Navigation Model

Use a responsive navigation system:

* Desktop: left sidebar navigation
* Tablet: collapsible sidebar
* Mobile: bottom navigation or slide-out drawer
* Top bar: global search, notifications, account menu, quick create action

## Desktop Layout Structure

```md
App Shell
├── Sidebar Navigation
├── Top Header
│   ├── Page Title
│   ├── Search
│   ├── Notifications
│   └── User Menu
└── Main Content
    ├── Page Header
    ├── Primary Content
    └── Contextual Panels / Drawers
```

## Mobile Layout Structure

```md
Mobile App Shell
├── Top Bar
│   ├── Menu
│   ├── Page Title
│   └── Quick Action
├── Main Content
└── Bottom Navigation
```

## Recommended Navigation Items

### Primary Navigation

* Dashboard
* Records
* Projects
* Tasks
* Analytics
* Team

### Secondary Navigation

* Notifications
* Help
* Settings
* Billing
* Integrations

### User Menu

* Profile
* Preferences
* Theme toggle
* Workspace switcher
* Sign out

## Content Hierarchy

Each main page should follow this order:

1. Page title and description
2. Primary action
3. Key summary metrics
4. Search and filters
5. Main data view
6. Contextual secondary content
7. Empty/loading/error state when applicable

***

# 5. Page-by-Page UI Specifications

## 5.1 Authentication Pages

### Pages

* Sign in
* Sign up
* Forgot password
* Reset password
* Verification screen

### Layout

Use a centered card layout on desktop and full-screen layout on mobile.

Desktop structure:

```md
Authentication Page
├── Brand Logo
├── Auth Card
│   ├── Heading
│   ├── Supporting Text
│   ├── Form Fields
│   ├── Primary Button
│   ├── Secondary Auth Options
│   └── Footer Link
└── Optional Visual Panel
```

### Visual Style

* Use generous whitespace
* Keep forms short and focused
* Optional right-side brand illustration for desktop
* Avoid heavy gradients
* Use subtle background pattern or soft radial glow

### Required Elements

* Logo
* Page heading
* Supporting description
* Email input
* Password input
* Primary submit button
* Alternative auth options if applicable
* Legal links
* Error messages

### Interaction Notes

* Show inline validation
* Disable submit button while loading
* Preserve entered email after failed login
* Provide clear error messaging
* Support password visibility toggle

***

## 5.2 Dashboard

### Purpose

The dashboard provides a high-level overview of the user's workspace, recent activity, priority items, and important metrics.

### Layout

```md
Dashboard
├── Page Header
│   ├── Greeting / Title
│   ├── Date or Context
│   └── Primary Action
├── Metrics Grid
├── Main Content Grid
│   ├── Activity Feed
│   ├── Priority Tasks
│   └── Recent Records
└── Optional Insights Panel
```

### Required Components

* Page heading
* Primary action button
* Metric cards
* Recent activity list
* Task summary
* Quick actions
* Empty state for new users

### Metric Cards

Each metric card should include:

* Label
* Value
* Trend indicator
* Supporting description
* Optional icon

Example metrics:

* Active projects
* Open tasks
* Completed this week
* Pending approvals
* Team activity

### Visual Direction

* Use a 12-column grid on desktop
* Use stacked cards on mobile
* Use clean cards with subtle borders
* Avoid overly colorful dashboard widgets
* Use accent color only for emphasis

***

## 5.3 Records List Page

### Purpose

The records page allows users to browse, search, filter, sort, select, and manage structured data.

### Layout

```md
Records Page
├── Page Header
│   ├── Title
│   ├── Description
│   └── Create Button
├── Toolbar
│   ├── Search
│   ├── Filters
│   ├── Sort
│   ├── View Toggle
│   └── Bulk Actions
├── Data Table / Card List
└── Pagination
```

### Required Elements

* Search input
* Filter button
* Sort control
* View switcher
* Table view for desktop
* Card view for mobile
* Row actions menu
* Pagination or infinite scroll
* Bulk selection

### Table Columns

Recommended columns:

* Name
* Status
* Owner
* Category
* Updated date
* Priority
* Actions

### Table Behavior

* Sticky header on long lists
* Sortable columns
* Row hover state
* Checkbox selection
* Contextual actions menu
* Keyboard accessible row actions
* Responsive collapse into cards on mobile

### Mobile Card Layout

Each card should include:

* Title
* Status badge
* Key metadata
* Last updated
* Primary action
* Overflow menu

***

## 5.4 Record Detail Page

### Purpose

The detail page gives users a complete view of a single record with metadata, activity, comments, and actions.

### Layout

```md
Record Detail
├── Header
│   ├── Breadcrumb
│   ├── Title
│   ├── Status
│   └── Actions
├── Main Content
│   ├── Overview Card
│   ├── Details Section
│   ├── Related Items
│   └── Activity Feed
└── Sidebar
    ├── Metadata
    ├── Owner
    ├── Dates
    └── Quick Actions
```

### Required Elements

* Breadcrumb navigation
* Record title
* Status badge
* Primary action
* Secondary actions menu
* Metadata sidebar
* Activity timeline
* Comments or notes section
* Related records

### Interaction Notes

* Use inline editing for simple fields
* Use modal or side panel for complex editing
* Confirm destructive actions
* Show autosave feedback where applicable
* Use skeleton loading for detail sections

***

## 5.5 Create / Edit Form Page

### Purpose

Allow users to create or update structured content with minimal friction.

### Layout

```md
Form Page
├── Page Header
│   ├── Title
│   ├── Description
│   └── Save Actions
├── Form Content
│   ├── Basic Information
│   ├── Details
│   ├── Assignment
│   └── Advanced Options
└── Sticky Footer
    ├── Cancel
    └── Save
```

### Form Design

* Group fields into logical sections
* Use clear labels above fields
* Provide helper text where needed
* Use inline validation
* Avoid multi-column forms on mobile
* Use sticky save actions for long forms

### Required Field Types

* Text input
* Textarea
* Select
* Multi-select
* Date picker
* Toggle
* Checkbox
* Radio group
* File upload
* Searchable combobox

### Validation

* Show validation after blur or submit
* Use clear, human-readable messages
* Do not rely on color alone
* Place error text near the relevant field
* Keep invalid fields accessible by screen reader

***

## 5.6 Analytics Page

### Purpose

The analytics page helps users understand trends, performance, and operational health.

### Layout

```md
Analytics Page
├── Page Header
│   ├── Title
│   ├── Description
│   └── Date Range Selector
├── KPI Cards
├── Chart Grid
│   ├── Primary Chart
│   ├── Secondary Chart
│   └── Breakdown Cards
└── Data Table
```

### Required Elements

* Date range picker
* KPI summary cards
* Line chart
* Bar chart
* Donut or progress chart
* Export button
* Filter controls
* Supporting data table

### Chart Design

* Keep charts minimal
* Avoid excessive colors
* Use accessible color contrast
* Include tooltips
* Include labels and legends
* Provide empty states when no data exists

### Visual Rules

* Use one primary accent color
* Use neutral grid lines
* Avoid 3D charts
* Use clear axis labels
* Ensure chart data is readable in dark mode

***

## 5.7 Settings Page

### Purpose

Allow users to manage account, workspace, preferences, security, billing, and integrations.

### Layout

```md
Settings Page
├── Settings Sidebar
├── Settings Content
│   ├── Section Header
│   ├── Setting Groups
│   └── Save Actions
```

### Settings Sections

* Profile
* Account
* Workspace
* Members
* Roles and permissions
* Notifications
* Appearance
* Security
* Billing
* Integrations
* API keys

### Visual Style

* Use simple section cards
* Group related controls
* Avoid dense configuration walls
* Use confirmation modals for risky changes
* Clearly label destructive actions

### Interaction Notes

* Save changes per section
* Show unsaved changes indicator
* Use toggles for binary preferences
* Use role badges for permissions
* Use copy buttons for API keys or IDs

***

## 5.8 Team Page

### Purpose

Allow users to view, invite, manage, and organize team members.

### Layout

```md
Team Page
├── Page Header
│   ├── Title
│   ├── Description
│   └── Invite Button
├── Toolbar
│   ├── Search
│   ├── Role Filter
│   └── Status Filter
├── Members Table
└── Pending Invitations
```

### Required Elements

* Team member avatar
* Name and email
* Role badge
* Status indicator
* Last active
* Actions menu
* Invite member modal
* Pending invites list

### Interaction Notes

* Invite flow should be modal-based
* Role changes should require confirmation if permissions are elevated
* Removing users should use a destructive confirmation modal

***

## 5.9 Notifications Page

### Purpose

Display user notifications and system events.

### Layout

```md
Notifications Page
├── Page Header
│   ├── Title
│   └── Mark All as Read
├── Filter Tabs
│   ├── All
│   ├── Unread
│   └── Mentions
└── Notification List
```

### Notification Item

Each item should include:

* Icon or avatar
* Title
* Description
* Timestamp
* Read/unread state
* Contextual action

### Interaction Notes

* Support marking individual notifications as read
* Support bulk mark all as read
* Use subtle unread indicators
* Avoid intrusive notification styling

***

# 6. Component System

## Buttons

### Variants

* Primary
* Secondary
* Tertiary
* Ghost
* Destructive
* Link
* Icon-only

### Button Sizes

* Small: 32px height
* Medium: 40px height
* Large: 48px height

### Button States

* Default
* Hover
* Active
* Focus
* Disabled
* Loading

### Button Style

```md
Primary Button:
- Background: Primary Color
- Text: White
- Border Radius: 12px
- Font Weight: 600
- Height: 40px
- Padding: 0 16px
```

## Inputs

### Input Types

* Text input
* Email input
* Password input
* Search input
* Textarea
* Select
* Combobox
* Date picker
* File upload

### Input States

* Default
* Hover
* Focus
* Filled
* Disabled
* Error
* Success

### Input Style

```md
Input Height: 40px
Input Radius: 12px
Input Border: 1px solid Border Color
Input Background: Surface Color
Focus Ring: 2px Primary Color at 30% opacity
Label Size: 14px
Helper Text Size: 13px
```

## Cards

### Card Types

* Standard card
* Metric card
* Interactive card
* Form section card
* Empty state card
* Chart card

### Card Style

```md
Card Background: Surface Color
Card Border: 1px solid Border Color
Card Radius: 16px
Card Padding: 24px
Card Shadow: Subtle shadow
```

## Navigation

### Sidebar

* Width: 264px
* Collapsed width: 72px
* Item height: 40px
* Item radius: 10px
* Icon size: 20px
* Label size: 14px

### Top Bar

* Height: 64px desktop
* Height: 56px mobile
* Contains page context, search, notifications, user menu

### Bottom Navigation

Use on mobile when there are 3 to 5 primary destinations.

Each item should include:

* Icon
* Label
* Active state
* Tap target minimum 44px

## Modals

### Modal Sizes

* Small: 400px
* Medium: 560px
* Large: 720px
* Full-screen on mobile

### Modal Structure

```md
Modal
├── Header
│   ├── Title
│   └── Close Button
├── Body
└── Footer
    ├── Secondary Action
    └── Primary Action
```

### Modal Behavior

* Trap focus
* Close with Escape
* Close with overlay click only for non-critical modals
* Require explicit cancel for destructive flows
* Return focus to triggering element after close

## Tables

### Table Requirements

* Sticky header
* Sortable columns
* Row hover state
* Checkbox selection
* Bulk action toolbar
* Empty state
* Loading skeleton
* Responsive card layout on mobile

### Table Row Height

```md
Compact Row: 44px
Default Row: 56px
Comfortable Row: 64px
```

## Badges

### Badge Types

* Status
* Priority
* Role
* Category
* Count

### Badge Style

```md
Badge Height: 24px
Badge Radius: 999px
Badge Padding: 0 10px
Badge Font Size: 12px
Badge Font Weight: 600
```

## Dropdown Menus

### Requirements

* Keyboard navigable
* Clear hover and focus states
* Support icons
* Support dividers
* Align to trigger
* Avoid opening off-screen
* Use destructive color for dangerous actions

## Toasts

### Toast Types

* Success
* Error
* Warning
* Info

### Toast Behavior

* Appear top-right on desktop
* Appear bottom on mobile
* Auto-dismiss after 4 to 6 seconds
* Provide close button
* Do not stack more than 3 visible toasts

***

# 7. Design Tokens

## Color Palette

### Light Mode

```md
Primary Color: #4F46E5
Primary Hover: #4338CA
Primary Active: #3730A3

Background: #F8FAFC
Surface: #FFFFFF
Surface Elevated: #FFFFFF
Surface Muted: #F1F5F9

Text Primary: #0F172A
Text Secondary: #475569
Text Tertiary: #64748B
Text Disabled: #94A3B8

Border: #E2E8F0
Border Strong: #CBD5E1

Success: #16A34A
Success Background: #DCFCE7

Warning: #D97706
Warning Background: #FEF3C7

Error: #DC2626
Error Background: #FEE2E2

Info: #2563EB
Info Background: #DBEAFE
```

### Dark Mode

```md
Primary Color: #818CF8
Primary Hover: #A5B4FC
Primary Active: #C7D2FE

Background: #0F172A
Surface: #111827
Surface Elevated: #1E293B
Surface Muted: #334155

Text Primary: #F8FAFC
Text Secondary: #CBD5E1
Text Tertiary: #94A3B8
Text Disabled: #64748B

Border: #334155
Border Strong: #475569

Success: #22C55E
Success Background: #052E16

Warning: #F59E0B
Warning Background: #451A03

Error: #F87171
Error Background: #450A0A

Info: #60A5FA
Info Background: #172554
```

## Typography

Use a modern sans-serif font stack.

```md
Font Family: Inter, SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif
```

### Typography Scale

```md
Display: 48px / 56px / 700
Heading 1: 36px / 44px / 700
Heading 2: 30px / 38px / 700
Heading 3: 24px / 32px / 650
Heading 4: 20px / 28px / 650
Body Large: 18px / 28px / 400
Body: 16px / 24px / 400
Body Small: 14px / 20px / 400
Caption: 12px / 16px / 500
Label: 14px / 20px / 600
Button: 14px / 20px / 600
```

## Spacing System

Use an 8px base spacing system.

```md
Spacing Unit: 8px

Space 0: 0px
Space 1: 4px
Space 2: 8px
Space 3: 12px
Space 4: 16px
Space 5: 20px
Space 6: 24px
Space 8: 32px
Space 10: 40px
Space 12: 48px
Space 16: 64px
Space 20: 80px
Space 24: 96px
```

## Border Radius

```md
Radius XS: 4px
Radius SM: 8px
Radius MD: 12px
Radius LG: 16px
Radius XL: 20px
Radius 2XL: 24px
Radius Full: 999px

Default Border Radius: 16px
Button Border Radius: 12px
Input Border Radius: 12px
Card Border Radius: 16px
Modal Border Radius: 20px
```

## Shadows

```md
Shadow XS: 0 1px 2px rgba(15, 23, 42, 0.06)
Shadow SM: 0 2px 6px rgba(15, 23, 42, 0.08)
Shadow MD: 0 8px 24px rgba(15, 23, 42, 0.10)
Shadow LG: 0 16px 40px rgba(15, 23, 42, 0.14)
Shadow XL: 0 24px 64px rgba(15, 23, 42, 0.18)
```

Dark mode shadows should be more subtle and may use black with reduced opacity.

```md
Dark Shadow MD: 0 8px 24px rgba(0, 0, 0, 0.32)
```

## Grid System

```md
Desktop Grid: 12 columns
Tablet Grid: 8 columns
Mobile Grid: 4 columns

Desktop Max Content Width: 1440px
Content Padding Desktop: 32px
Content Padding Tablet: 24px
Content Padding Mobile: 16px

Grid Gap Desktop: 24px
Grid Gap Tablet: 20px
Grid Gap Mobile: 16px
```

## Z-Index Scale

```md
Base: 0
Dropdown: 100
Sticky Header: 200
Overlay: 400
Modal: 500
Toast: 600
Tooltip: 700
```

***

# 8. Responsive Behavior

## Breakpoints

```md
Mobile: 0px - 639px
Tablet: 640px - 1023px
Desktop: 1024px - 1439px
Large Desktop: 1440px+
```

## Mobile Behavior

* Use single-column layouts
* Convert tables into stacked cards
* Use full-screen modals
* Collapse sidebar into drawer
* Show bottom navigation for primary sections
* Hide non-essential metadata behind disclosure sections
* Keep primary action visible
* Use large tap targets of at least 44px

## Tablet Behavior

* Use two-column layouts where appropriate
* Sidebar may collapse to icons
* Modals can remain centered
* Tables may horizontally scroll only when necessary
* Cards should use 2-column grids

## Desktop Behavior

* Use persistent sidebar navigation
* Use multi-column layouts
* Use right-side contextual panels
* Use full data tables
* Keep toolbar actions visible
* Support keyboard shortcuts and advanced interactions

## Large Desktop Behavior

* Increase content breathing room
* Do not stretch forms beyond readable width
* Cap main content at a max width
* Use side panels for contextual information
* Maintain readable line lengths

## Content Width Rules

```md
Readable Text Max Width: 720px
Form Max Width: 760px
Dashboard Max Width: 1440px
Settings Content Max Width: 960px
Modal Body Max Width: 720px
```

***

# 9. Interaction Patterns

## Primary Actions

Each page should have one clear primary action.

Examples:

* Create record
* Invite member
* Save changes
* Export report
* Add task

Primary actions should appear in the page header or sticky footer depending on context.

## Secondary Actions

Secondary actions should be visually quieter and placed near the relevant content.

Examples:

* Filter
* Sort
* Duplicate
* Archive
* Download
* Share

## Destructive Actions

Destructive actions require:

* Red visual treatment
* Confirmation modal
* Clear explanation of consequences
* Explicit action label

Example destructive labels:

* Delete record
* Remove member
* Revoke access
* Cancel subscription

Avoid vague labels like "Confirm" for destructive actions.

## Search

Search should be available in data-heavy areas.

Search behavior:

* Debounce input
* Show loading indicator
* Highlight matching results when useful
* Provide empty search result state
* Allow clear action
* Preserve filters when searching

## Filters

Filters should support:

* Status
* Owner
* Date range
* Category
* Priority
* Tags

Filter behavior:

* Show active filter count
* Allow clearing individual filters
* Allow clearing all filters
* Persist filters when navigating back
* Use drawer-based filters on mobile

## Sorting

Sorting should be available for lists and tables.

Recommended sort options:

* Recently updated
* Created date
* Name
* Priority
* Status

## Bulk Actions

Bulk actions appear only after selecting items.

Bulk action toolbar should include:

* Selected count
* Common actions
* Clear selection
* Destructive action if relevant

## Inline Editing

Use inline editing for low-risk fields.

Examples:

* Title
* Description
* Status
* Assignee
* Tags

Show save feedback immediately.

## Confirmation Patterns

Use confirmations for:

* Deleting
* Removing users
* Revoking access
* Publishing irreversible changes
* Leaving a form with unsaved changes

## Keyboard Behavior

Required keyboard support:

* Tab navigation
* Enter to activate primary controls
* Escape to close modals and menus
* Arrow keys for menus and lists
* Command/Ctrl + K for global search
* Command/Ctrl + S for save where appropriate

***

# 10. Motion & Animation

## Motion Principles

Motion should be:

* Fast
* Subtle
* Purposeful
* Consistent
* Non-distracting

Avoid dramatic bounces, excessive parallax, or long transitions.

## Duration Tokens

```md
Instant: 75ms
Fast: 120ms
Base: 180ms
Slow: 240ms
Extra Slow: 320ms
```

## Easing Tokens

```md
Ease Out: cubic-bezier(0.16, 1, 0.3, 1)
Ease In: cubic-bezier(0.7, 0, 0.84, 0)
Ease In Out: cubic-bezier(0.65, 0, 0.35, 1)
Spring Soft: 220ms ease-out
```

## Recommended Animations

### Page Transitions

* Fade in content
* Slight upward movement of 4px to 8px
* Duration: 180ms

### Modal Entry

* Fade overlay
* Scale modal from 96% to 100%
* Duration: 180ms

### Drawer Entry

* Slide from right on desktop
* Slide from bottom on mobile
* Duration: 240ms

### Dropdown Entry

* Fade and scale from 98% to 100%
* Duration: 120ms

### Button Interaction

* Hover background transition
* Active scale: 0.98
* Duration: 120ms

### Loading Skeleton

* Use subtle shimmer
* Duration: 1200ms to 1600ms
* Avoid high-contrast shimmer in dark mode

## Reduced Motion

If the user prefers reduced motion:

* Disable non-essential transitions
* Remove scale effects
* Keep simple opacity changes
* Avoid shimmer animations
* Preserve functional state changes

***

# 11. Accessibility Requirements

## WCAG Target

The interface must meet WCAG AA standards.

## Color Contrast

Minimum contrast requirements:

* Normal text: 4.5:1
* Large text: 3:1
* UI components: 3:1
* Focus indicators: clearly visible against background

## Keyboard Accessibility

All interactive elements must be reachable and usable by keyboard.

Required:

* Visible focus states
* Logical tab order
* No keyboard traps except intentional modal focus trap
* Escape closes modals, drawers, popovers, and menus
* Enter and Space activate buttons and menu items

## Focus States

Use consistent focus rings.

```md
Focus Ring: 2px solid Primary Color
Focus Ring Offset: 2px
Focus Ring Radius: Matches component radius
```

## Screen Reader Support

Required:

* Semantic landmarks
* Proper heading hierarchy
* Form labels connected to inputs
* Error messages announced
* Buttons have descriptive labels
* Icon-only buttons include accessible labels
* Tables include column headers
* Modals announce title and description

## Touch Accessibility

* Minimum tap target: 44px by 44px
* Do not place destructive actions too close to primary actions
* Ensure swipe gestures have non-gesture alternatives

## Forms Accessibility

* Labels must always be visible
* Do not rely on placeholder text as label
* Required fields must be indicated clearly
* Error messages must be connected to fields
* Use clear language for validation

## Data Visualization Accessibility

Charts should include:

* Text summary
* Accessible labels
* Tooltips
* Data table alternative where possible
* Patterns or labels in addition to color

***

# 12. Empty / Loading / Error States

## Empty States

Empty states should be helpful, calm, and action-oriented.

### Empty State Structure

```md
Empty State
├── Icon or Illustration
├── Title
├── Description
├── Primary Action
└── Optional Secondary Action
```

### Empty State Example

```md
Title: No records yet
Description: Create your first record to start organizing your workspace.
Primary Action: Create record
Secondary Action: Import data
```

## Search Empty State

```md
Title: No results found
Description: Try adjusting your search or clearing filters.
Primary Action: Clear filters
```

## Loading States

Use skeleton loaders rather than spinners for content areas.

Recommended loading patterns:

* Skeleton cards for dashboards
* Skeleton rows for tables
* Skeleton form sections
* Button spinner for submit actions
* Page-level progress only for long operations

## Error States

Error states should explain what happened and what the user can do next.

### Error State Structure

```md
Error State
├── Icon
├── Title
├── Description
├── Retry Action
└── Secondary Help Link
```

### Error State Example

```md
Title: Something went wrong
Description: We couldn’t load this data. Check your connection and try again.
Primary Action: Try again
Secondary Action: Contact support
```

## Form Errors

* Show errors inline
* Place error below field
* Use red color and icon
* Explain how to fix the issue
* Move focus to first invalid field after submit

## Permission Error

```md
Title: You don’t have access
Description: Contact your workspace admin if you believe this is a mistake.
Primary Action: Go back
```

## Offline State

```md
Title: You’re offline
Description: Changes will sync when your connection is restored.
Status: Sync pending
```

## Success Feedback

Use success feedback sparingly.

Examples:

* Toast after save
* Inline "Saved" indicator
* Progress completion
* Confirmation screen after major action

***

# 13. AI Generation Notes

## Overall Visual Direction

Generate a premium SaaS interface with a clean, modern, minimal aesthetic. The product should feel comparable in quality to Linear, Notion, Stripe, Vercel, Framer, Airbnb, and Apple.

Prioritize:

* Clean spacing
* Strong visual hierarchy
* Subtle borders
* Soft shadows
* Rounded corners
* Clear typography
* Balanced layouts
* Minimal color usage
* Strong light and dark mode support

## Style Keywords

Use these design keywords:

```md
Modern SaaS
Premium dashboard
Minimal interface
Clean data layout
Subtle depth
Soft shadows
Rounded cards
Elegant typography
Calm colors
High contrast
Accessible UI
Mobile-first
Production-ready
```

## Layout Guidance

Use:

* 12-column desktop grid
* 8-column tablet grid
* 4-column mobile grid
* Persistent sidebar on desktop
* Collapsible drawer on mobile
* Top header with search and actions
* Card-based content sections
* Clean tables for structured data

## Component Guidance

All generated screens should use reusable components:

* App shell
* Sidebar
* Top bar
* Page header
* Metric cards
* Data table
* Search bar
* Filter controls
* Status badges
* Buttons
* Inputs
* Modals
* Drawers
* Toasts
* Empty states

## Visual Constraints

Avoid:

* Overly saturated colors
* Heavy gradients
* Glassmorphism as the primary style
* Excessive shadows
* Cluttered dashboards
* Tiny text
* Poor contrast
* Decorative animations
* Inconsistent spacing
* Random color usage
* Complex illustrations unless needed

## Dark Mode Notes

Dark mode should feel native, not inverted.

Use:

* Deep slate background
* Slightly lighter elevated surfaces
* Muted borders
* Soft primary accent
* High-contrast text
* Reduced shadow intensity
* Clear focus rings

## Content Tone

Use concise, professional, human-centered language.

Good examples:

```md
Create record
Invite member
View details
Save changes
No results found
Try adjusting your filters
Your changes have been saved
```

Avoid:

```md
Submit
Click here
Oops!
Invalid operation
Something bad happened
```

## Screen Density

The product should support professional workflows but remain breathable.

Use:

* Comfortable default spacing
* Compact table option only where useful
* Progressive disclosure for advanced fields
* Secondary panels for metadata
* Drawers for contextual editing

***

# 14. Technical Frontend Notes

## Recommended Frontend Stack

```md
Framework: React or Next.js
Styling: Tailwind CSS
Component Architecture: Reusable design system components
Icons: Lucide, Heroicons, or similar clean icon set
Charts: Recharts, Visx, or similar accessible charting library
Forms: React Hook Form or equivalent
Validation: Zod or equivalent schema validation
Tables: TanStack Table or equivalent
Animation: Framer Motion or CSS transitions
```

## Component Architecture

Recommended structure:

```md
components/
├── app-shell/
├── navigation/
├── buttons/
├── forms/
├── cards/
├── tables/
├── modals/
├── drawers/
├── charts/
├── feedback/
├── empty-states/
└── layout/
```

## Layout Architecture

Use an app shell pattern:

```md
AppShell
├── Sidebar
├── Header
├── Main
└── ToastProvider
```

## State Management

Use local state for simple UI interactions.

Use shared state for:

* Authentication
* Workspace context
* Theme
* User preferences
* Filters
* Selected records
* Notifications

## Theming

Implement theme using CSS variables.

Required theme variables:

```md
--color-background
--color-surface
--color-surface-elevated
--color-text-primary
--color-text-secondary
--color-border
--color-primary
--color-error
--color-success
--radius-card
--radius-button
--shadow-card
```

## Dark Mode Implementation

Dark mode should be controlled by:

* System preference by default
* User override in preferences
* Persistent local storage value
* Class-based or data-attribute theme switching

Example behavior:

```md
Default: system
Options: light, dark, system
Persistence: local storage or user profile setting
```

## Responsive Implementation

Use mobile-first CSS.

Recommended approach:

```md
Base styles: mobile
sm: tablet adjustments
lg: desktop layout
xl: large desktop optimization
```

## Data Table Implementation

Tables should support:

* Sorting
* Filtering
* Pagination
* Row selection
* Bulk actions
* Column visibility
* Loading skeleton
* Empty state
* Responsive card fallback

## Form Implementation

Forms should include:

* Client-side validation
* Server-side error handling
* Accessible error messages
* Disabled loading state
* Dirty state tracking
* Unsaved changes warning
* Keyboard submit behavior

## Performance Requirements

Prioritize:

* Fast initial load
* Lazy-loaded heavy routes
* Code-splitting
* Optimized images
* Memoized table rows
* Virtualized long lists where needed
* Avoid unnecessary re-renders

## Accessibility Implementation

Ensure:

* Semantic HTML
* ARIA only when necessary
* Focus management for modals and drawers
* Skip-to-content link
* Keyboard navigable menus
* Accessible icon buttons
* Form labels and descriptions
* Screen-reader-friendly validation

## Production Readiness Checklist

```md
- Responsive across mobile, tablet, and desktop
- Supports light and dark mode
- Meets WCAG AA contrast
- Has loading states
- Has empty states
- Has error states
- Uses reusable components
- Uses consistent tokens
- Handles long content gracefully
- Handles missing data gracefully
- Supports keyboard navigation
- Includes focus states
- Avoids layout shift
- Optimizes performance
- Provides clear user feedback
```
