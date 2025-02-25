#   B m a d - a g e n t 
 
 A   R A G   ( R e t r i e v a l   A u g m e n t e d   G e n e r a t i o n )   p i p e l i n e   f o r   a n s w e r i n g   q u e s t i o n s   a b o u t   t h e   B m a d   c h a r g e d   p a r t i c l e   s i m u l a t i o n   l i b r a r y   u s i n g   t h e   B m a d   m a n u a l   a s   a   k n o w l e d g e   s o u r c e . 
 
 # #   S e t u p 
 
 1 .   I n s t a l l   d e p e n d e n c i e s : 
 
 ` ` ` b a s h 
 p i p   i n s t a l l   o p e n a i   l a n g c h a i n   l a n g c h a i n _ h u g g i n g f a c e   f a i s s - c p u 
 ` ` ` 
 
 2 .   S e t   y o u r   O p e n A I   A P I   k e y : 
 
 ` ` ` b a s h 
 e x p o r t   O P E N A I _ A P I _ K E Y = " y o u r - a p i - k e y - h e r e " 
 ` ` ` 
 
 # #   U s a g e 
 
 # # #   I n t e r a c t i v e   M o d e 
 
 R u n   t h e   a g e n t   i n   i n t e r a c t i v e   m o d e   t o   a s k   q u e s t i o n s   a b o u t   B m a d : 
 
 ` ` ` b a s h 
 p y t h o n   b m a d _ a g e n t . p y 
 ` ` ` 
 
 # # #   S i n g l e   Q u e r y   M o d e 
 
 ` ` ` b a s h 
 p y t h o n   b m a d _ a g e n t . p y   " H o w   d o e s   q u a d r u p o l e   t r a c k i n g   w o r k   i n   B m a d ? " 
 ` ` ` 
 
 # # #   A d d i t i o n a l   O p t i o n s 
 
 ` ` ` b a s h 
 p y t h o n   b m a d _ a g e n t . p y   - - m o d e l   g p t - 4 o   - - v e r b o s e   " W h a t   i s   a   g r o u p   e l e m e n t   i n   B m a d ? " 
 ` ` ` 
 
 # #   F e a t u r e s 
 
 -   V e c t o r   d a t a b a s e   b u i l t   f r o m   t h e   B m a d   m a n u a l   d o c u m e n t a t i o n 
 -   R e t r i e v a l   o f   r e l e v a n t   c o n t e x t   b a s e d   o n   s e m a n t i c   s e a r c h 
 -   I n t e g r a t i o n   w i t h   O p e n A I   m o d e l s   f o r   a n s w e r i n g   q u e s t i o n s 
 -   I n t e r a c t i v e   c o m m a n d - l i n e   i n t e r f a c e 
 
 # #   P r o j e c t   S t r u c t u r e 
 
 -   ` c l e a n . p y ` :   P r e p r o c e s s i n g   s c r i p t   f o r   c l e a n i n g   L a T e X   f i l e s 
 -   ` b u i l d _ d b . i p y n b ` :   N o t e b o o k   f o r   b u i l d i n g   t h e   F A I S S   v e c t o r   d a t a b a s e 
 -   ` b m a d _ a g e n t . p y ` :   M a i n   q u e r y   i n t e r f a c e   f o r   t h e   R A G   p i p e l i n e 
 -   ` b m a d _ d o c / ` :   O r i g i n a l   B m a d   m a n u a l   f i l e s 
 -   ` c l e a n _ b m a d _ d o c / ` :   P r o c e s s e d   t e x t   f i l e s 
 -   ` f a i s s _ t e x / ` :   V e c t o r   d a t a b a s e   f i l e s 